# 📦 08_rosbag_processing: ROS2 Bag Ingestion & Sensor Synchronization

This module encapsulates all core utilities for reading, indexing, time-synchronizing, and decoding recorded ROS2 bags containing RGB-D camera, 2D LiDAR laser scans, and mobile robot odometry/TF chains.

---

## 🛠️ Modules & Purpose

1. **`rosbag_rgbd_sim_capture.py`**:
   - Opens ROS2 bags (`AnyReader` with `ROS2_HUMBLE` typestore).
   - Indexes and buffers stream topics:
     * `/camera/rgb/image_rect_color` (RGB 2D color images)
     * `/camera/depth_registered/image_raw` (Registered depth images `32FC1` / `16UC1`)
     * `/camera/rgb/camera_info` (Camera pinhole intrinsics $f_x, f_y, c_x, c_y$)
     * `/tf` & `/tf_static` (Robot kinematic transforms to world/odom frame)
     * `/odom` (Wheel odometry fallback)
     * `/scan` (2D LiDAR planar laser range measurements)
   - Performs nearest-timestamp synchronization across RGB and depth streams within strict temporal thresholds.
   - De-projects depth maps and 2D laser scans into 3D metric coordinate frames.

2. **`rgbd_bag_processing.py`**:
   - Processing loop for streaming sequential frames into object detection models.
   - Decoupled from mapping logic to allow pure detector benchmarking.

---

## 💡 Typical Usage in Mapping Pipeline

```python
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import rosbag_rgbd_sim_capture as capture

typestore = get_typestore(Stores.ROS2_HUMBLE)
with AnyReader([bag_path], default_typestore=typestore) as reader:
    topic_map, stats = capture._topic_message_map(reader, selected_topics)
```
