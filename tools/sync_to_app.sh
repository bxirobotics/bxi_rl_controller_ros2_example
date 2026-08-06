sudo rm -rf /opt/bxi/bxi_rl_controller_ros2_example
colcon build --merge-install --install-base /opt/bxi/bxi_rl_controller_ros2_example --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -Wall -Wextra -Wpedantic