#!/bin/bash

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
        echo "Please source this file: source ./setup_env.sh" >&2
        exit 1
fi

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP=/opt/ros/humble/setup.bash
BXI_SETUP=/opt/bxi/bxi_ros2_pkg/setup.bash
LOCAL_SETUP="$WORKSPACE_DIR/install/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
        echo "Missing ROS 2 setup file: $ROS_SETUP" >&2
        return 1
fi

if [ ! -f "$BXI_SETUP" ]; then
        echo "Missing BXI ROS 2 setup file: $BXI_SETUP" >&2
        return 1
fi

if [ ! -f "$LOCAL_SETUP" ]; then
        echo "Missing local setup file: $LOCAL_SETUP" >&2
        echo "Run ./build.sh first." >&2
        return 1
fi

source "$ROS_SETUP"
source "$BXI_SETUP"
source "$LOCAL_SETUP"
