#!/bin/bash
set -e

ROS_SETUP=/opt/ros/humble/setup.bash
BXI_SETUP=/opt/bxi/bxi_ros2_pkg/setup.bash

if [ ! -f "$ROS_SETUP" ]; then
        echo "Missing ROS 2 setup file: $ROS_SETUP" >&2
        exit 1
fi

if [ ! -f "$BXI_SETUP" ]; then
        echo "Missing BXI ROS 2 setup file: $BXI_SETUP" >&2
        exit 1
fi

source "$ROS_SETUP"
source "$BXI_SETUP"

# Set the default build type
# BUILD_TYPE=RelWithDebInfo
BUILD_TYPE=Release
colcon build \
        --merge-install \
        --cmake-args "-DCMAKE_BUILD_TYPE=$BUILD_TYPE" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On"\
        -Wall -Wextra -Wpedantic


# --symlink-install \
