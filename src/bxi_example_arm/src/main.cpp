#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "control/RobotController.h"

using namespace rclcpp::executors;

int main(int argc, char** argv) {

    rclcpp::init(argc, argv);
    MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);

    auto robotControllerPtr = std::make_shared<RobotController>();

    try {
        robotControllerPtr->init();

        executor.add_node(robotControllerPtr);
        executor.spin();
    } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(robotControllerPtr->get_logger(), e.what() << '\n');
    }
    rclcpp::shutdown();
    return 0;
}