#include "bxi_depth_camera/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<bxi_depth_camera::CameraManager>();
        rclcpp::executors::MultiThreadedExecutor executor(
            rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
        executor.remove_node(node);
        node.reset();
    } catch (const std::exception &error) {
        RCLCPP_FATAL(rclcpp::get_logger("depth_camera_manager"), "%s",
                     error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
