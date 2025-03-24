#!/bin/bash
set -e

# Set the default build type
# BUILD_TYPE=RelWithDebInfo
BUILD_TYPE=Release
colcon build \
        --merge-install \
        --cmake-args "-DCMAKE_BUILD_TYPE=$BUILD_TYPE" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" -DCMAKE_CXX_FLAGS="-DUSE_SLC=1"\
        -Wall -Wextra -Wpedantic


# --symlink-install \