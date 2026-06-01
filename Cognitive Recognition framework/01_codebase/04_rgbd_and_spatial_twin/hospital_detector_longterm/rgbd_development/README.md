# RGB-D Development Folder Structure

This folder contains all RGB-D spatial memory development assets organized by purpose.

## Folder Layout

```
rgbd_development/
├── scripts/                    # Python scripts
│   ├── rgbd_spatial_twin.py    # Main RGB-D viewer and exporter
│   ├── rgbd_hospitalguard_detect.py # RGB-only detection entrypoint for TUM exports
│   ├── hospitalguard_temporal_core.py # Local copy of temporal ByteTrack core
│   ├── rgbd_hospitalguard_temporal.py # Unified RGB-D + Temporal(ByteTrack) pipeline
│   └── hospital_twin.db_init.py # Database initialization utility
├── data/                       # RGB-D datasets (junction/symlinks to extracted TUM data)
│   ├── rgbd_dataset_freiburg1_xyz/
│   └── rgbd_dataset_freiburg3_long_office_household/
├── output/                     # Exported results
│   ├── hospital_twin.db        # SQLite spatial memory database
│   ├── exports/                # RGB and RGB-D exports
│   ├── detections/             # HospitalGuard annotated videos
│   └── logs/                   # RGB-D-specific Excel logs
└── README.md                   # This file
```

## Quick Start

### View RGB-D Stream (Freiburg1)
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_spatial_twin.py --sequence-root "..\data\rgbd_dataset_freiburg1_xyz" --no-db --wait-ms 15
```

### Export to Video (Freiburg1)
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_spatial_twin.py --sequence-root "..\data\rgbd_dataset_freiburg1_xyz" --no-db --output-video "..\output\exports\rgbd_freiburg1_replay.mp4"
```

### Export to Video (Freiburg3)
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_spatial_twin.py --sequence-root "..\data\rgbd_dataset_freiburg3_long_office_household" --no-db --output-video "..\output\exports\rgbd_freiburg3_replay.mp4"
```

### View with Spatial Memory Logging
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_spatial_twin.py --sequence-root "..\data\rgbd_dataset_freiburg1_xyz" --db "..\output\hospital_twin.db" --max-frames 500
```

### Run HospitalGuard Detection With RGB-D-Specific Storage
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_hospitalguard_detect.py --input-video "..\output\exports\rgb_freiburg1_export.mp4"
```

### Run Unified RGB-D + HospitalGuard Temporal (ByteTrack)
```powershell
cd hospital_detector_longterm\rgbd_development\scripts
python rgbd_hospitalguard_temporal.py --sequence-root "..\data\rgbd_dataset_freiburg1_xyz"
```

This unified script writes all outputs only under this folder:
- `../output/detections/` (annotated video)
- `../output/logs/` (CSV + Excel)
- `../output/hospital_twin.db` (SQLite spatial memory)

## Command-Line Options

- `--sequence-root`: Path to TUM RGB-D dataset folder (required)
- `--db`: SQLite database path (default: ../output/hospital_twin.db)
- `--no-db`: Viewer mode; do not write points to SQLite
- `--output-video`: MP4 file path to export side-by-side RGB-depth video
- `--max-frames`: Cap for number of frames to replay
- `--wait-ms`: Delay between frames in ms for viewer mode
- `--max-time-diff`: Max RGB-depth timestamp difference in seconds (default: 0.02)
- `--online-url`: URL to download a TUM RGB-D sequence

`rgbd_hospitalguard_detect.py` writes:
- annotated videos to `../output/detections/`
- Excel logs to `../output/logs/hospitalguard_rgbd_log.xlsx`

`rgbd_hospitalguard_temporal.py` writes:
- annotated videos to `../output/detections/`
- CSV logs to `../output/logs/`
- Excel logs to `../output/logs/hospitalguard_temporal_rgbd_log.xlsx`
- tracked 3D object memory to `../output/hospital_twin.db`

## Controls

- **q** or **Esc**: Exit viewer

## Data Files

Both datasets are already extracted in `../Data/`:
- `rgbd_dataset_freiburg1_xyz`: ~427 MB (standard office environment)
- `rgbd_dataset_freiburg3_long_office_household`: ~1.4 GB (longer office sequence)

The `data/` folder contains directory junctions (symlinks) to these extracted datasets for easy reference.
