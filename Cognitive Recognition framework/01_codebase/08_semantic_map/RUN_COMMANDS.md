# 🗺️ 3D Semantic Mapping & Digital Twin — Run Commands

This guide provides the exact commands to run the 3D Cognitive Recognition framework on **any recorded ROS2 RGB-D bag**.

---

## ⚡ 1. Run Full 3D Reconstruction on Any Bag

Run this command from the **project root folder**:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\08_semantic_map\build_semantic_map_from_bag.py `
  --bag "02_datasets\kmi\rgbd_pose_20261103_143812" `
  --frame-stride 4 `
  --min-consensus-hits 3
```

> **To run on another bag:** Simply replace `--bag "<PATH_TO_YOUR_BAG_FOLDER>"`.  
> Output artifacts are automatically saved to: `04_outputs_runs_and_logs/outputs/semantic_maps/<bag_name>/`

---

## 🔄 2. Fast Replay / Re-generate Without Re-running Inference

If you already processed a bag and only want to replay the geometry, camera trajectory, and point cloud into Rerun without re-running YOLO/SAM/DINO models:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\08_semantic_map\build_semantic_map_from_bag.py `
  --bag "02_datasets\kmi\rgbd_pose_20261103_143812" `
  --rerun-only
```

---

## 🖥️ 3. View in Rerun 3D Desktop Viewer

Open the complete canonical 3D Digital Twin (`world_map.rrd`):
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\rerun.exe `
  04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\world_map.rrd
```

*Includes: Environmental 3D RGB-D point cloud, 2D LiDAR corridor perimeter walls, 3D Green/Red Keep-Clear door egress cylinders, exact landmark labels & physical surface points (zero cuboids), and timeline-scrubbable synchronized video.*

---

## 🌐 4. View in Interactive 3D Web Browser (HTML)

Open the interactive 3D Web Viewer with Three.js egress cylinders and video tab:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\08_semantic_map\view_semantic_map.py `
  --db 04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\world_map.db `
  --html
```

Or open the generated file directly in your web browser:
```powershell
Start-Process "04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\semantic_map_ontology.html"
```

---

## 📦 5. Standalone Canonical RRD Exporter

To re-export the lightweight canonical `world_map.rrd` directly from any existing `world_map.db`:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\08_semantic_map\export_clean_rrd.py `
  --db 04_outputs_runs_and_logs\outputs\semantic_maps\rgbd_pose_20261103_143812\world_map.db
```

---

## 📁 Output Artifacts Reference

Each run produces the following artifacts in `04_outputs_runs_and_logs/outputs/semantic_maps/<bag_name>/`:

| File | Description |
| :--- | :--- |
| `world_map.rrd` | Complete 3D Rerun Digital Twin (Point Cloud, LiDAR, Video Stream, Egress Cylinders, No Cuboids). |
| `semantic_map_ontology.html` | Interactive 3D Web viewer with Three.js cylinders, QSR scene graph, and video stream. |
| `world_map.db` | SQLite database storing all 3D consensus landmarks, camera poses, and navigable zones. |
| `hospital_compliance_report.json` | Automated Health & Safety compliance evaluation report. |
| `preview.mp4` | Annotated video recording with YOLO detections, FastSAM masks, and OCR text badges. |
| `environment_point_cloud.npz` | Cached structural 3D point cloud for instant RRD re-exports. |

---

## ⚙️ Essential Command Flags

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--bag` | *(required)* | Path to the ROS2 bag directory. |
| `--frame-stride` | `1` | Frame skip stride (`4` processes $4\times$ faster with full spatial coverage). |
| `--min-consensus-hits` | `1` | Multi-view hit threshold for general objects (`3` recommended; signs enforce $\ge 10$). |
| `--reset-map` | `False` | Wipe existing persistent database and rebuild completely from scratch. |
| `--rerun-only` | `False` | Geometry & cloud replay only, preserving landmarks without re-running models. |
| `--rerun-spawn` | `False` | Automatically pop up the live Rerun desktop window during bag processing. |
| `--no-preview` | `False` | Disable the OpenCV live preview window (recommended for headless execution). |
| `--no-video` | `False` | Skip writing `preview.mp4`. |
