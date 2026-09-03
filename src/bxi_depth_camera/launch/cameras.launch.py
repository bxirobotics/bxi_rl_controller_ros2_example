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
    "enable_sync": bool,
    "enable_rgbd": bool,
    "imu_sync_method": str,
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
    "depth_qos": str,
    "depth_info_qos": str,
    "color_qos": str,
    "color_info_qos": str,
    "infra1_qos": str,
    "infra1_info_qos": str,
    "infra2_qos": str,
    "infra2_info_qos": str,
    "gyro_qos": str,
    "accel_qos": str,
    "pointcloud.pointcloud_qos": str,
    "rgbd_qos": str,
    "linear_accel_cov": float,
    "angular_velocity_cov": float,
    "rgb_camera.enable_auto_exposure": bool,
    "rgb_camera.exposure": float,
    "rgb_camera.gain": float,
    "depth_module.enable_auto_exposure": bool,
    "depth_module.exposure": float,
    "depth_module.gain": float,
    "auto_exposure_roi.left": int,
    "auto_exposure_roi.top": int,
    "auto_exposure_roi.right": int,
    "auto_exposure_roi.bottom": int,
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
    "threshold_filter.enable": bool,
    "threshold_filter.min_distance": float,
    "threshold_filter.max_distance": float,
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
