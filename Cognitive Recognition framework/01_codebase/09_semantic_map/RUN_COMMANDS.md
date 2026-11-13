# 🗺️ 3D Semantic Mapping & Digital Twin — Run Commands

This guide provides the commands to run the 3D Cognitive Recognition framework and open the 3D Digital Twin on **any recorded ROS2 RGB-D bag**.

All commands below are run from the **project root folder**:
```powershell
cd "D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework"
```

---

## ⚡ 1. Run Full 3D Reconstruction on Any Bag

### Option A — Interactive Numbered Menu (Easiest)
Simply run:
```powershell
.\run_bag.bat
```
This scans `02_datasets/` and displays a numbered menu of all available ROS2 bags (`[1]`, `[2]`, `[3]`, …). Type the bag number and press **Enter**.

### Option B — Short Bag Keyword / Timestamp
Pass any unique part of the bag name (or menu number):
```powershell
.\run_bag.bat 153314
.\run_bag.bat 143812
```
*(You can also pass extra flags, e.g. `.\run_bag.bat 153314 --no-preview` or drag-and-drop a bag folder onto `run_bag.bat` in Windows Explorer.)*

### Option C — Direct Python CLI
`build_semantic_map_from_bag.py` also supports omitting `--bag` (for the interactive menu), passing a short keyword (`--bag 153314`), or passing a full path:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\09_semantic_map\build_semantic_map_from_bag.py `
  --bag 153314 `
  --frame-stride 4 `
  --min-consensus-hits 3 `
  --reset-map
```

---

## 🖥️ 2. View in Rerun 3D Desktop Viewer (`world_map.rrd`)

### Option A — Interactive Map Menu
```powershell
.\view_map.bat
```
Displays all generated maps sorted newest-first with `.rrd` sizes. Press **Enter** to open the newest map or type a number (`1`, `2`, …).

### Option B — Open by Short Bag Keyword / Timestamp
```powershell
.\view_map.bat 143812
.\view_map.bat 153314
```

### Option C — Direct Rerun Executable
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\rerun.exe `
  04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\world_map.rrd
```

---

## 🌐 3. View in Interactive 3D Web Browser (`semantic_map_ontology.html`)

### Option A — Interactive Menu
```powershell
.\view_map.bat --html
```

### Option B — Open by Short Keyword
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\09_semantic_map\view_semantic_map.py `
  --db 143812 `
  --html
```

---

## 🔄 4. Fast Replay / Re-export Canonical RRD Without Re-running Models

To re-export `world_map.rrd` directly from an existing `world_map.db`:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\09_semantic_map\export_clean_rrd.py `
  --db 04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\world_map.db
```

---

## 📁 Output Artifacts Reference

Each run saves the following artifacts in `04_outputs_runs_and_logs/outputs/semantic_maps/<bag_name>/`:

| File | Description |
| :--- | :--- |
| `world_map.rrd` | Complete 3D Rerun Digital Twin (Point Cloud, LiDAR, Video Stream, Egress Zones, Landmarks). |
| `semantic_map_ontology.html` | Interactive 3D Web viewer with QSR scene graph, H&S audit, and video stream. |
| `world_map.db` | SQLite database storing all 3D consensus landmarks, camera poses, and navigable zones. |
| `hospital_compliance_report.md` | Automated Health & Safety compliance evaluation report (Markdown + JSON). |
| `preview.mp4` | Annotated video recording with detections, FastSAM masks, and safety sign badges. |
| `environment_point_cloud.npz` | Cached structural 3D point cloud for instant RRD re-exports. |
