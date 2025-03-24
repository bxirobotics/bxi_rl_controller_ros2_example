#include "control/RobotController.h"
#include <rcpputils/asserts.hpp>

RobotController::RobotController() : Node("RobotController")
{
    this->declare_parameter("/topic_prefix", "");

    topic_prefix_ = this->get_parameter("/topic_prefix")
                        .get_parameter_value()
                        .get<std::string>();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);
    actuators_cmds_pub_ptr_ =
        this->create_publisher<communication::msg::ActuatorCmds>(
            topic_prefix_ + "actuators_cmds", qos);

    joints_state_subscription_ =
        this->create_subscription<sensor_msgs::msg::JointState>(
            topic_prefix_ + "joint_states", qos,
            std::bind(&RobotController::joint_callback, this,
                      std::placeholders::_1));

    imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
        topic_prefix_ + "imu_data", qos,
        std::bind(&RobotController::imu_callback, this, std::placeholders::_1));

    odom_subscription_ =
        this->create_subscription<nav_msgs::msg::Odometry>(
            topic_prefix_ + "odom", qos,
            std::bind(&RobotController::odom_callback, this,
                      std::placeholders::_1));

    touch_subscription_ =
        this->create_subscription<communication::msg::TouchSensor>(
            topic_prefix_ + "touch_sensor", qos,
            std::bind(&RobotController::touch_callback, this,
                      std::placeholders::_1));

    reset_state_client_ =
        this->create_client<communication::srv::RobotReset>(
            topic_prefix_ + "robot_reset");
}

void RobotController::reset_robot(int reset_step, bool release)
{

    auto request =
        std::make_shared<communication::srv::RobotReset::Request>();

    request->header.frame_id = std::string("bot_elf");
    request->reset_step = reset_step;
    request->release = release;

    RCLCPP_INFO(this->get_logger(), "waiting for service %s ...",
                reset_state_client_->get_service_name());
    while (!reset_state_client_->wait_for_service(20ms))
    {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Interrupted while waiting for the service. Exiting.");
            return;
        }
    }

    auto result = reset_state_client_->async_send_request(request);

    if (result.get()->is_success)
    {
        RCLCPP_INFO(this->get_logger(), "call service reset_state success");
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to reset state");
    }
}

void RobotController::imu_callback(
    const sensor_msgs::msg::Imu::SharedPtr msg) const
{
    // todo
    (void)msg;
}

void RobotController::joint_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) const
{
    // todo
    (void)msg;
}

void RobotController::touch_callback(
    const communication::msg::TouchSensor::SharedPtr msg) const
{
    // todo
    (void)msg;
}

//only for simulation
void RobotController::odom_callback(
    const nav_msgs::msg::Odometry::SharedPtr msg) const
{
    // todo
    (void)msg;
}

void RobotController::init()
{
    inner_loop_thread_ = std::thread(&RobotController::inner_loop, this);
    run_ = true;
}

RobotController::~RobotController()
{
    run_ = false;
    inner_loop_thread_.join();
}

void RobotController::inner_loop()
{
    rclcpp::Rate loop_rate(1.0 / dt_);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::string joint_name[] = {
        "l_hip_z_joint",
        "l_hip_x_joint",
        "l_hip_y_joint",
        "l_knee_y_joint",
        "l_ankle_y_joint",
        "l_ankle_x_joint",

        "r_hip_z_joint",
        "r_hip_x_joint",
        "r_hip_y_joint",
        "r_knee_y_joint",
        "r_ankle_y_joint",
        "r_ankle_x_joint",

        "l_shld_y_joint",
        "l_shld_x_joint",
        "l_shld_z_joint",
        "l_elb_y_joint",

        "r_shld_y_joint",
        "r_shld_x_joint",
        "r_shld_z_joint",
        "r_elb_y_joint"};

    double joint_pos[] = {
        0,
        0.04,
        -0.5,
        1.0,
        -0.5,
        -0.04,

        0,
        -0.04,
        -0.5,
        1.0,
        -0.5,
        0.04,

        0.7,
        0.2,
        -0.1,
        -1.5,

        0.7,
        -0.2,
        0.1,
        -1.5};

    double joint_kp[] =
        {
            25,
            25,
            30,
            40,
            3,
            3,

            25,
            25,
            30,
            40,
            3,
            3,

            10,
            10,
            10,
            10,

            10,
            10,
            10,
            10,
        };

    double joint_kd[] =
        {
            2.5,
            2.5,
            3,
            4,
            0.3,
            0.3,

            2.5,
            2.5,
            3,
            4,
            0.3,
            0.3,

            1,
            1,
            1,
            1,

            1,
            1,
            1,
            1,
        };

    reset_robot(1, false); // Enable motor position control

    // init motor pos
    while (rclcpp::ok() && run_)
    {
        static size_t loop_count = 0;
        static double soft_start = 0;

        communication::msg::ActuatorCmds actuatorCmds;
        actuatorCmds.header.frame_id = std::string("bot_elf");
        actuatorCmds.header.stamp = this->now();

        for (size_t i = 0; i < sizeof(joint_name) / sizeof(joint_name[0]); i++)
        {
            actuatorCmds.actuators_name.emplace_back(joint_name[i]);
            actuatorCmds.pos.emplace_back(joint_pos[i]);
            actuatorCmds.vel.emplace_back(0);
            actuatorCmds.torque.emplace_back(0);

            soft_start = loop_count / (1. / dt_);
            soft_start = loop_count > 1 ? 1 : soft_start;

            actuatorCmds.kp.emplace_back(joint_kp[i] * soft_start);
            actuatorCmds.kd.emplace_back(joint_kd[i]);
        }

        actuators_cmds_pub_ptr_->publish(actuatorCmds);

        loop_count++;
        if (loop_count > (10. / dt_)) //hold for 10s
        {
            break;
        }

        loop_rate.sleep();
    }

    // release robt(only for simulation), enable motor velocity and torque control
    reset_robot(2, true);

    // run your control algorithm
    while (rclcpp::ok() && run_)
    {
        static size_t loop_count = 0;
        communication::msg::ActuatorCmds actuatorCmds;
        actuatorCmds.header.frame_id = std::string("bot_elf");
        actuatorCmds.header.stamp = this->now();

        for (size_t i = 0; i < sizeof(joint_name) / sizeof(joint_name[0]); i++)
        {
            actuatorCmds.actuators_name.emplace_back(joint_name[i]);
            actuatorCmds.pos.emplace_back(joint_pos[i]);
            actuatorCmds.vel.emplace_back(0);
            actuatorCmds.torque.emplace_back(0);

            actuatorCmds.kp.emplace_back(joint_kp[i]);
            actuatorCmds.kd.emplace_back(joint_kd[i]);
        }

        actuators_cmds_pub_ptr_->publish(actuatorCmds);

        loop_count++;

        loop_rate.sleep();
    }
}