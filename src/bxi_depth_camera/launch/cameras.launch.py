from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_USE_CONFIG = "__use_config__"

ARGUMENTS = {
    "camera_namespace": str,
    "serial_no": str,
    "single_camera_name": str,
    "discovery_interval_sec": float,
    "retry_interval_sec": float,
    "device_timeout_sec": float,
    "enable_depth": bool,
    "enable_color": bool,
    "enable_infra1": bool,
    "enable_infra2": bool,
    "enable_gyro": bool,
    "enable_accel": bool,
    "align_depth.enable": bool,
    "pointcloud.enable": bool,
    "pointcloud.ordered_pc": bool,
    "pointcloud.allow_no_texture_points": bool,
    "pointcloud.max_fps": float,
    "depth_module.depth_profile": str,
    "depth_module.rectification.enable": bool,
    "rgb_camera.color_profile": str,
    "rgb_camera.rectification.enable": bool,
    "infra1.rectification.enable": bool,
    "infra2.rectification.enable": bool,
    "decimation_filter.enable": bool,
    "decimation_filter.filter_magnitude": int,
    "spatial_filter.enable": bool,
    "spatial_filter.filter_smooth_alpha": float,
    "spatial_filter.filter_smooth_delta": float,
    "spatial_filter.holes_fill": int,
    "temporal_filter.enable": bool,
    "temporal_filter.filter_smooth_alpha": float,
    "temporal_filter.filter_smooth_delta": float,
    "temporal_filter.holes_fill": int,
    "hole_filling_filter.enable": bool,
    "hole_filling_filter.holes_fill": int,
    "second_hole_filling_filter.enable": bool,
    "second_hole_filling_filter.holes_fill": int,
    "orbbec.enable_sdk_filters": bool,
    "orbbec.fallback_hfov": float,
    "orbbec.fallback_vfov": float,
}


def _camera_node(context):
    parameters = {
        name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
        for name, value_type in ARGUMENTS.items()
        if LaunchConfiguration(name).perform(context) != _USE_CONFIG
    }
    default_config = PathJoinSubstitution(
        [FindPackageShare("bxi_depth_camera"), "config", "default.yaml"]
    )
    parameter_sources = [default_config]
    config_file = LaunchConfiguration("config_file").perform(context).strip()
    if config_file:
        parameter_sources.append(config_file)
    if parameters:
        parameter_sources.append(parameters)
    return [
        Node(
            package="bxi_depth_camera",
            executable="cameras",
            output="screen",
            emulate_tty=True,
            parameters=parameter_sources,
        )
    ]


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=_USE_CONFIG)
        for name in ARGUMENTS
    ]
    return LaunchDescription(
        [
            *declared,
            DeclareLaunchArgument("config_file", default_value=""),
            OpaqueFunction(function=_camera_node),
        ]
    )
