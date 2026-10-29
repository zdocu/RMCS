#include <cmath>

#include <algorithm>
#include <array>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <fmt/format.h>
#include <rclcpp/node.hpp>
#include <rmcs_description/tf_description.hpp>
#include <rmcs_executor/component.hpp>
#include <rmcs_utility/eigen_structured_bindings.hpp>

#include "controller/chassis/qcp_solver.hpp"
#include "controller/pid/matrix_pid_calculator.hpp"
#include "controller/pid/pid_calculator.hpp"

namespace rmcs_core::controller::chassis {

class DeformableSteeringWheelController
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    DeformableSteeringWheelController()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true))
        , mass_(get_parameter("mass").as_double())
        , moment_of_inertia_(get_parameter("moment_of_inertia").as_double())
        , wheel_radius_(get_parameter("wheel_radius").as_double())
        , friction_coefficient_(get_parameter("friction_coefficient").as_double())
        , k1_(get_parameter("k1").as_double())
        , k2_(get_parameter("k2").as_double())
        , no_load_power_(get_parameter("no_load_power").as_double())
        , com_height_(get_parameter_or("com_height", 0.0))
        , chassis_radius_x_(get_parameter_or("chassis_radius_x", 0.2341741 / std::numbers::sqrt2))
        , chassis_radius_y_(get_parameter_or("chassis_radius_y", 0.2341741 / std::numbers::sqrt2))
        , dt_(get_parameter_or("dt", 0.001))
        , steering_angle_kp_(get_parameter_or("steering_angle_kp", 8.0))
        , steering_angle_ki_(get_parameter_or("steering_angle_ki", 0.0))
        , steering_angle_kd_(get_parameter_or("steering_angle_kd", 0.0))
        , steering_angle_velocity_limit_(get_parameter_or("steering_angle_velocity_limit", 12.0))
        , steering_velocity_kp_(get_parameter_or("steering_velocity_kp", 0.35))
        , steering_velocity_ki_(get_parameter_or("steering_velocity_ki", 0.0))
        , steering_velocity_kd_(get_parameter_or("steering_velocity_kd", 0.0))
        , steering_torque_limit_(get_parameter_or("steering_torque_limit", 3.0))
        , chassis_radius_base_(get_parameter_or("chassis_radius", 0.2341741))
        , rod_length_(get_parameter_or("rod_length", 0.140))
        , deform_rate_filter_(get_parameter_or("deform_rate_filter", 0.5))
        , radius_rate_limit_(get_parameter_or("radius_rate_limit", 0.25))
        , deform_speed_limit_(get_parameter_or("deform_speed_limit", 0.35))
        , deform_speed_threshold_(get_parameter_or("deform_speed_threshold", 0.004))
        , translational_velocity_pid_calculator_(5.0, 0.0, 0.0)
        , angular_velocity_pid_calculator_(5.0, 0.0, 0.0)
        , wheel_velocity_pid_(0.6, 0.0, 0.0) {

        register_input("/chassis/left_front_wheel/max_torque", wheel_motor_max_control_torque_);

        for (size_t i = 0; i < kWheelCount; ++i) {
            register_input(
                fmt::format("/chassis/{}_wheel/velocity", kWheelName[i]), wheel_velocity_[i]);
            register_output(
                fmt::format("/chassis/{}_wheel/control_torque", kWheelName[i]),
                wheel_control_torque_[i], nan_);

            register_input(
                fmt::format("/chassis/{}_steering/angle", kWheelName[i]), steering_angle_[i]);
            register_input(
                fmt::format("/chassis/{}_steering/velocity", kWheelName[i]), steering_velocity_[i]);
            register_output(
                fmt::format("/chassis/{}_steering/control_torque", kWheelName[i]),
                steering_control_torque_[i], nan_);

            register_input(
                fmt::format("/chassis/{}_joint/physical_angle", kWheelName[i]), joint_angle_[i]);
            register_input(
                fmt::format("/chassis/{}_joint/physical_velocity", kWheelName[i]),
                joint_velocity_[i]);

            steering_angle_pid_[i].kp = steering_angle_kp_;
            steering_angle_pid_[i].ki = steering_angle_ki_;
            steering_angle_pid_[i].kd = steering_angle_kd_;
            steering_angle_pid_[i].output_min = -steering_angle_velocity_limit_;
            steering_angle_pid_[i].output_max = steering_angle_velocity_limit_;

            steering_velocity_pid_[i].kp = steering_velocity_kp_;
            steering_velocity_pid_[i].ki = steering_velocity_ki_;
            steering_velocity_pid_[i].kd = steering_velocity_kd_;
            steering_velocity_pid_[i].output_min = -steering_torque_limit_;
            steering_velocity_pid_[i].output_max = steering_torque_limit_;
        }

        register_input("/chassis/control_velocity", chassis_control_velocity_);
        register_input("/chassis/control_power_limit", power_limit_);
        register_input("/chassis/radius", chassis_radius_);

        register_output("/chassis/steering_aligned", steering_aligned_, false);
        register_output("/chassis/deform/rate", deform_rate_, 0.0);
        register_output("/chassis/deform/active", deform_active_, false);
    }

    void update() override {
        if (std::isnan(chassis_control_velocity_->vector[0])) {
            reset_all_controls();
            return;
        }

        const double radius = *chassis_radius_;
        if (!std::isfinite(radius) || radius <= 1e-6) {
            reset_all_controls();
            return;
        }

        update_deform_state(radius);
        const Eigen::Vector4d module_radial_rate = calculate_module_radial_rate();

        Eigen::Vector4d wheel_velocities;
        Eigen::Vector4d steering_angles;
        Eigen::Vector4d steering_velocities;
        for (size_t i = 0; i < kWheelCount; ++i) {
            wheel_velocities[i] = *wheel_velocity_[i];
            steering_angles[i] = *steering_angle_[i];
            steering_velocities[i] = *steering_velocity_[i];
        }

        Eigen::Vector3d control_velocity = chassis_control_velocity_->vector;
        if (*deform_active_)
            control_velocity.head<2>() =
                clamp_norm(control_velocity.head<2>(), static_cast<double>(deform_speed_limit_));

        const auto [module_angles, module_speeds] =
            calculate_module_reference(control_velocity, module_radial_rate);

        auto [aligned_angles, aligned_speeds] =
            align_steering(module_angles, module_speeds, steering_angles);

        update_steering_control(aligned_angles, steering_angles, steering_velocities);

        const auto chassis_velocity =
            calculate_chassis_velocity(wheel_velocities, steering_angles, module_radial_rate);

        auto chassis_control_torque = calculate_chassis_control_torque(chassis_velocity);
        const auto wheel_pid_torques =
            calculate_wheel_pid_torques(wheel_velocities, aligned_speeds);
        chassis_control_torque.torque = constrain_chassis_control_torque(
            wheel_velocities, steering_angles, chassis_control_torque, wheel_pid_torques);
        const auto wheel_control_torques = calculate_wheel_control_torques(
            chassis_control_torque, wheel_pid_torques, steering_angles);
        for (size_t i = 0; i < kWheelCount; ++i)
            *wheel_control_torque_[i] = wheel_control_torques[i];
    }

private:
    static constexpr size_t kWheelCount = 4;
    static constexpr const char* kWheelName[] = {
        "left_front",
        "left_back",
        "right_back",
        "right_front",
    };
    static constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();
    static constexpr double g_ = 9.81;

    struct ChassisControlTorque {
        Eigen::Vector2d torque;
        Eigen::Vector2d lambda;
    };

    static Eigen::Vector2d clamp_norm(const Eigen::Vector2d& value, double limit) {
        const double norm = value.norm();
        if (norm > limit && norm > 1e-9)
            return value * (limit / norm);
        return value;
    }

    void reset_all_controls() {
        for (size_t i = 0; i < kWheelCount; ++i) {
            *wheel_control_torque_[i] = 0.0;
            *steering_control_torque_[i] = 0.0;
            steering_angle_pid_[i].reset();
            steering_velocity_pid_[i].reset();
        }
        *steering_aligned_ = false;
        *deform_rate_ = 0.0;
        module_radial_rate_.setZero();
        *deform_active_ = false;
    }

    void update_deform_state(double /*radius*/) {
        double sum = 0.0;
        for (size_t i = 0; i < kWheelCount; ++i) {
            const double alpha = *joint_angle_[i];
            const double alpha_dot = *joint_velocity_[i];
            if (!std::isfinite(alpha) || !std::isfinite(alpha_dot)) {
                module_radial_rate_[i] = 0.0;
                continue;
            }
            module_radial_rate_[i] = -rod_length_ * std::sin(alpha) * alpha_dot;
            sum += module_radial_rate_[i];
        }

        const double raw_rate = sum / static_cast<double>(kWheelCount);
        filtered_radius_rate_ =
            (1.0 - deform_rate_filter_) * filtered_radius_rate_ + deform_rate_filter_ * raw_rate;
        filtered_radius_rate_ =
            std::clamp(filtered_radius_rate_, -radius_rate_limit_, radius_rate_limit_);
        if (std::abs(filtered_radius_rate_) < deform_speed_threshold_)
            filtered_radius_rate_ = 0.0;

        *deform_rate_ = filtered_radius_rate_;
        *deform_active_ = std::abs(filtered_radius_rate_) > deform_speed_threshold_;
    }

    double module_radius(size_t i) const {
        const double alpha = *joint_angle_[i];
        if (!std::isfinite(alpha))
            return std::max(*chassis_radius_, 1e-6);
        return chassis_radius_base_ + rod_length_ * std::cos(alpha);
    }

    double mean_module_radius() const {
        double sum = 0.0;
        for (size_t i = 0; i < kWheelCount; ++i)
            sum += module_radius(i);
        return sum / static_cast<double>(kWheelCount);
    }

    Eigen::Vector2d module_position(size_t i) const {
        const double a = module_radius(i) / std::numbers::sqrt2;
        switch (i) {
        case 0: return {-a, +a};
        case 1: return {-a, -a};
        case 2: return {+a, -a};
        case 3: return {+a, +a};
        default: return {0.0, 0.0};
        }
    }

    Eigen::Vector2d module_radial_direction(size_t i) const {
        const double norm = module_position(i).norm();
        if (norm < 1e-9)
            return Eigen::Vector2d::UnitX();
        return module_position(i) / norm;
    }

    Eigen::Vector4d calculate_module_radial_rate() const { return module_radial_rate_; }

    std::pair<Eigen::Vector4d, Eigen::Vector4d> calculate_module_reference(
        const Eigen::Vector3d& control_velocity, const Eigen::Vector4d& radial_rate) const {
        const Eigen::Vector2d linear = control_velocity.head<2>();
        const double angular = control_velocity[2];

        Eigen::Vector4d angles;
        Eigen::Vector4d speeds;
        for (size_t i = 0; i < kWheelCount; ++i) {
            const Eigen::Vector2d position = module_position(i);
            const Eigen::Vector2d radial = module_radial_direction(i);

            Eigen::Vector2d velocity =
                linear + angular * Eigen::Vector2d{-position.y(), position.x()};
            velocity += radial_rate[i] * radial;

            const double speed = velocity.norm();
            if (speed < 1e-9) {
                angles[i] = *steering_angle_[i];
                speeds[i] = 0.0;
            } else {
                angles[i] = std::atan2(velocity.y(), velocity.x());
                speeds[i] = speed / wheel_radius_;
            }
        }
        return {angles, speeds};
    }

    std::pair<Eigen::Vector4d, Eigen::Vector4d> align_steering(
        Eigen::Vector4d angles, Eigen::Vector4d speeds,
        const Eigen::Vector4d& current_angles) const {
        for (size_t i = 0; i < kWheelCount; ++i) {
            const double flipped = wrap_to_pi(angles[i] + std::numbers::pi);
            const double delta_direct = std::abs(wrap_to_pi(angles[i] - current_angles[i]));
            const double delta_flipped = std::abs(wrap_to_pi(flipped - current_angles[i]));
            if (delta_flipped < delta_direct) {
                angles[i] = flipped;
                speeds[i] = -speeds[i];
            }
        }
        return {angles, speeds};
    }

    void update_steering_control(
        const Eigen::Vector4d& target_angles, const Eigen::Vector4d& current_angles,
        const Eigen::Vector4d& current_velocities) {
        bool aligned = true;
        for (size_t i = 0; i < kWheelCount; ++i) {
            const double position_error = wrap_to_pi(target_angles[i] - current_angles[i]);
            const double velocity_reference = steering_angle_pid_[i].update(position_error);
            const double velocity_error = velocity_reference - current_velocities[i];
            const double torque = steering_velocity_pid_[i].update(velocity_error);

            *steering_control_torque_[i] =
                std::clamp(torque, -steering_torque_limit_, steering_torque_limit_);

            if (std::abs(position_error) > steering_aligned_tolerance_)
                aligned = false;
        }
        *steering_aligned_ = aligned;
    }

    Eigen::Vector3d calculate_chassis_velocity(
        const Eigen::Vector4d& wheel_velocities, const Eigen::Vector4d& steering_angles,
        const Eigen::Vector4d& radial_rate) const {
        Eigen::MatrixXd a(kWheelCount, 3);
        Eigen::Vector4d b;
        for (size_t i = 0; i < kWheelCount; ++i) {
            const Eigen::Vector2d position = module_position(i);
            const Eigen::Vector2d radial = module_radial_direction(i);
            const Eigen::Vector2d direction{
                std::cos(steering_angles[i]), std::sin(steering_angles[i])};
            a(i, 0) = direction.x();
            a(i, 1) = direction.y();
            a(i, 2) = direction.x() * (-position.y()) + direction.y() * position.x();
            b[i] = wheel_velocities[i] * wheel_radius_ - radial_rate[i] * direction.dot(radial);
        }
        return (a.transpose() * a).ldlt().solve(a.transpose() * b);
    }

    ChassisControlTorque calculate_chassis_control_torque(const Eigen::Vector3d& chassis_velocity) {
        ChassisControlTorque result;

        Eigen::Vector3d chassis_velocity_error =
            chassis_control_velocity_->vector - chassis_velocity;
        Eigen::Vector2d translational_torque =
            (-std::numbers::sqrt2 / 4 * wheel_radius_) * mass_
            * translational_velocity_pid_calculator_.update(chassis_velocity_error.head<2>());
        result.torque.x() = translational_torque.norm();

        const double a_plus_b = std::numbers::sqrt2 * std::max(mean_module_radius(), 1e-6);
        result.torque.y() = (-std::numbers::sqrt2 / 4 * wheel_radius_)
                          * (moment_of_inertia_ / a_plus_b)
                          * angular_velocity_pid_calculator_.update(chassis_velocity_error[2]);

        if (result.torque.x() > 0)
            result.lambda = translational_torque / result.torque.x();
        else
            result.lambda = Eigen::Vector2d::UnitX();

        return result;
    }

    Eigen::Vector4d calculate_wheel_pid_torques(
        const Eigen::Vector4d& wheel_velocities, const Eigen::Vector4d& wheel_reference) {
        return wheel_velocity_pid_.update(wheel_reference - wheel_velocities);
    }

    Eigen::Vector2d constrain_chassis_control_torque(
        const Eigen::Vector4d& wheel_velocities, const Eigen::Vector4d& steering_angles,
        const ChassisControlTorque& chassis_control_torque,
        const Eigen::Vector4d& wheel_pid_torques) const {
        const auto& [w1, w2, w3, w4] = wheel_velocities;

        const auto& [x_max, y_max] = chassis_control_torque.torque;
        const double y_sign = y_max > 0 ? 1.0 : -1.0;

        const Eigen::Vector2d lambda_left{
            (std::cos(steering_angles[0]) + std::cos(steering_angles[1])) / 2.0,
            (std::sin(steering_angles[0]) + std::sin(steering_angles[1])) / 2.0};
        const Eigen::Vector2d lambda_right{
            (std::cos(steering_angles[2]) + std::cos(steering_angles[3])) / 2.0,
            (std::sin(steering_angles[2]) + std::sin(steering_angles[3])) / 2.0};

        const double l_left = lambda_left.norm() > 1e-6 ? lambda_left.norm() : 1e-6;
        const double l_right = lambda_right.norm() > 1e-6 ? lambda_right.norm() : 1e-6;
        const std::array<double, 2> lambda = {lambda_left.y() / l_left, lambda_right.y() / l_right};

        const auto& [t1, t2, t3, t4] = wheel_pid_torques;

        const double rhombus_top = (friction_coefficient_ * mass_ * g_ * wheel_radius_) / 4;
        const double rhombus_right =
            rhombus_top / std::max(std::abs(lambda[0]), std::abs(lambda[1]));

        const double a = 4 * k1_;
        const double b = 0;
        const double c = 4 * k1_;
        const double d = lambda[0] * (w1 - w3 + 2 * k1_ * (t1 - t3))
                       + lambda[1] * (w2 - w4 + 2 * k1_ * (t2 - t4));
        const double e = y_sign * (2 * k1_ * wheel_pid_torques.sum() + wheel_velocities.sum());
        const double f = k1_ * wheel_pid_torques.array().pow(2).sum()
                       + (wheel_pid_torques.array() * wheel_velocities.array()).sum()
                       + k2_ * wheel_velocities.array().pow(2).sum() - no_load_power_
                       - *power_limit_;

        Eigen::Vector2d result = Eigen::Vector2d::Constant(nan_);
        if (com_height_ > 1e-6) {
            const double dir_x = -(lambda[0] + lambda[1]) / 2.0;
            const double dir_y = (lambda[0] - lambda[1]) / 2.0;
            const double coeff = -com_height_ / (std::numbers::sqrt2 * wheel_radius_);
            const double gamma_1 = coeff * (+dir_x / chassis_radius_x_ + dir_y / chassis_radius_y_);
            const double gamma_2 = coeff * (-dir_x / chassis_radius_x_ + dir_y / chassis_radius_y_);

            const double force_to_torque = friction_coefficient_ * wheel_radius_;
            const double rhs = force_to_torque * mass_ * g_ / 4.0;
            const std::vector<QcpSolver::HalfPlaneConstraint> half_planes = {
                {lambda[0] - force_to_torque * gamma_1, y_sign, rhs},
                {-lambda[0] - force_to_torque * gamma_1, -y_sign, rhs},
                {lambda[1] - force_to_torque * gamma_2, y_sign, rhs},
                {-lambda[1] - force_to_torque * gamma_2, -y_sign, rhs},
                {-lambda[0] + force_to_torque * gamma_1, y_sign, rhs},
                {lambda[0] + force_to_torque * gamma_1, -y_sign, rhs},
                {-lambda[1] + force_to_torque * gamma_2, y_sign, rhs},
                {lambda[1] + force_to_torque * gamma_2, -y_sign, rhs},
            };
            result = qcp_solver_.solve(
                {1.0, 1.0}, {x_max, std::abs(y_max)}, half_planes, {a, b, c, d, e, f});
        }

        if (!result.array().isFinite().all()) {
            result = qcp_solver_.solve(
                {1.0, 1.0}, {x_max, std::abs(y_max)}, {rhombus_right, rhombus_top},
                {a, b, c, d, e, f});
        }
        result.y() *= y_sign;
        return result;
    }

    Eigen::Vector4d calculate_wheel_control_torques(
        const ChassisControlTorque& chassis_control_torque,
        const Eigen::Vector4d& wheel_pid_torques, const Eigen::Vector4d& steering_angles) const {
        const Eigen::Vector2d& resultant = chassis_control_torque.lambda;
        const double resultant_norm = resultant.norm();
        const Eigen::Vector2d unit = resultant_norm > 1e-6
                                       ? Eigen::Vector2d(resultant / resultant_norm)
                                       : Eigen::Vector2d::UnitX();

        Eigen::Vector4d wheel_torques;
        for (size_t i = 0; i < kWheelCount; ++i) {
            const Eigen::Vector2d direction{
                std::cos(steering_angles[i]), std::sin(steering_angles[i])};
            const double projection = direction.dot(unit);
            wheel_torques[i] =
                chassis_control_torque.torque.x() * projection + chassis_control_torque.torque.y();
        }
        wheel_torques.array() += wheel_pid_torques.array();
        return wheel_torques;
    }

    static double wrap_to_pi(double angle) {
        while (angle > std::numbers::pi)
            angle -= 2 * std::numbers::pi;
        while (angle < -std::numbers::pi)
            angle += 2 * std::numbers::pi;
        return angle;
    }

    const double mass_;
    const double moment_of_inertia_;
    const double wheel_radius_;
    const double friction_coefficient_;
    const double k1_, k2_, no_load_power_;
    const double com_height_;
    const double chassis_radius_x_;
    const double chassis_radius_y_;
    const double dt_;

    const double steering_angle_kp_, steering_angle_ki_, steering_angle_kd_;
    const double steering_angle_velocity_limit_;
    const double steering_velocity_kp_, steering_velocity_ki_, steering_velocity_kd_;
    const double steering_torque_limit_;
    const double chassis_radius_base_;
    const double rod_length_;
    const double deform_rate_filter_;
    const double radius_rate_limit_;
    const double deform_speed_limit_;
    const double deform_speed_threshold_;

    static constexpr double steering_aligned_tolerance_ = 0.06;

    InputInterface<double> wheel_motor_max_control_torque_;

    std::array<InputInterface<double>, kWheelCount> wheel_velocity_;
    std::array<OutputInterface<double>, kWheelCount> wheel_control_torque_;

    std::array<InputInterface<double>, kWheelCount> joint_angle_;
    std::array<InputInterface<double>, kWheelCount> joint_velocity_;
    std::array<InputInterface<double>, kWheelCount> steering_angle_;
    std::array<InputInterface<double>, kWheelCount> steering_velocity_;
    std::array<OutputInterface<double>, kWheelCount> steering_control_torque_;

    InputInterface<rmcs_description::BaseLink::DirectionVector> chassis_control_velocity_;
    InputInterface<double> power_limit_;
    InputInterface<double> chassis_radius_;

    OutputInterface<bool> steering_aligned_;
    OutputInterface<double> deform_rate_;
    OutputInterface<bool> deform_active_;

    pid::MatrixPidCalculator<2> translational_velocity_pid_calculator_;
    pid::PidCalculator angular_velocity_pid_calculator_;
    pid::MatrixPidCalculator<4> wheel_velocity_pid_;
    std::array<pid::PidCalculator, kWheelCount> steering_angle_pid_;
    std::array<pid::PidCalculator, kWheelCount> steering_velocity_pid_;

    QcpSolver qcp_solver_;

    double filtered_radius_rate_ = 0.0;
    Eigen::Vector4d module_radial_rate_ = Eigen::Vector4d::Zero();
    bool radius_initialized_ = false;
};

} // namespace rmcs_core::controller::chassis

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
    rmcs_core::controller::chassis::DeformableSteeringWheelController, rmcs_executor::Component)