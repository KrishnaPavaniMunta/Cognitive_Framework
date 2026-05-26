"""
infer_hospitalguard_longterm.py  —  Long-Term Memory Object Tracking
─────────────────────────────────────────────────────────────────────────────
HospitalGuard-109  —  interactive inference with Grounding DINO fallback
					   and ByteTrack long-term memory object tracking.

 (Migrated from infer_hospitalguard_temporal.py, May 2026)
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import tempfile
import requests
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
ROOT_DIR   = BASE_DIR.parent   # yolo_tr/ — where outputs/ and models/ live
V1_PATH    = ROOT_DIR / "outputs/runs/hospital/phase2_neck_head/weights/best.pt"
V3_PATH    = ROOT_DIR / "outputs/runs/hospital_v3/phase2_neck_head/weights/best.pt"
OUT_DIR    = ROOT_DIR / "outputs/hospitalguard_output"
EXCEL_PATH = ROOT_DIR / "outputs/hospitalguard_log.xlsx"
OUTPUT_RUN_TAG = "upd_motion_stab_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ...existing code from infer_hospitalguard_temporal.py continues unchanged...
