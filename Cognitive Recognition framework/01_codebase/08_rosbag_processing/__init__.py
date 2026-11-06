"""
08_rosbag_processing: ROS2 RGB-D bag capture, topic synchronization, and frame extraction.
"""
from __future__ import annotations

from .rosbag_rgbd_sim_capture import (
    CameraIntrinsics,
    DEFAULT_CAMERA_INFO_TOPIC,
    DEFAULT_DEPTH_TOPIC,
    DEFAULT_ODOM_TOPIC,
    DEFAULT_RGB_TOPIC,
    DEFAULT_TF_TOPIC,
    ExtrinsicsRecord,
    TimedMessage,
    _choose_extrinsics,
    _closest_index,
    _decode_color_image,
    _decode_depth_image_mm,
    _intrinsics_from_camera_info,
    _topic_message_map,
    project_laserscan_to_world,
)

__all__ = [
    "CameraIntrinsics",
    "DEFAULT_CAMERA_INFO_TOPIC",
    "DEFAULT_DEPTH_TOPIC",
    "DEFAULT_ODOM_TOPIC",
    "DEFAULT_RGB_TOPIC",
    "DEFAULT_TF_TOPIC",
    "ExtrinsicsRecord",
    "TimedMessage",
    "_choose_extrinsics",
    "_closest_index",
    "_decode_color_image",
    "_decode_depth_image_mm",
    "_intrinsics_from_camera_info",
    "_topic_message_map",
    "project_laserscan_to_world",
]
