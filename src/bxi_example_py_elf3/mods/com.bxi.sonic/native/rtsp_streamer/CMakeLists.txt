cmake_minimum_required(VERSION 3.16)
project(sonic_head_camera_rtsp LANGUAGES CXX)

# The Mod process inherits the sourced ROS environment. Avoid embedding the
# developer's Conda/overlay paths in the portable executable.
set(CMAKE_SKIP_RPATH TRUE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavcodec
  libavformat
  libavutil
  libswscale
)

add_executable(head_camera_rtsp_node head_camera_rtsp_node.cpp)
target_compile_features(head_camera_rtsp_node PRIVATE cxx_std_17)
target_compile_options(head_camera_rtsp_node PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
ament_target_dependencies(head_camera_rtsp_node rclcpp sensor_msgs)
target_link_libraries(head_camera_rtsp_node PkgConfig::FFMPEG)
