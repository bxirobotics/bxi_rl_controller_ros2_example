#pragma once

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <communication/msg/touch_sensor.hpp>
#include <communication/srv/robot_reset.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <communication/msg/actuator_cmds.hpp>
#include <communication/msg/motion_commands.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace rclcpp;
using namespace std::chrono_literals;

class RobotController : public Node
{
public:
    RobotController();

    ~RobotController();

    void init();

private:
    void inner_loop();

    void commands_callback(const communication::msg::MotionCommands::SharedPtr msg) const;

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) const;

    void touch_callback(const communication::msg::TouchSensor::SharedPtr msg) const;

    void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg) const;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) const;

    void reset_robot(int reset_step, bool release);

private:
    std::string topic_prefix_;

    rclcpp::Publisher<communication::msg::ActuatorCmds>::SharedPtr
        actuators_cmds_pub_ptr_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    rclcpp::Subscription<communication::msg::TouchSensor>::SharedPtr
        touch_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
        joints_state_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;

    std::thread inner_loop_thread_;

    Client<communication::srv::RobotReset>::SharedPtr reset_state_client_;

    double dt_ = 0.002;
    bool run_ = false;
};
