"""Unit test for MultimodalSignDetector (EasyOCR + CLIP)."""

import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODEBASE_DIR = PROJECT_ROOT / "01_codebase" / "07_object_detection"
if str(CODEBASE_DIR) not in sys.path:
    sys.path.insert(0, str(CODEBASE_DIR))

from multimodal_sign_detector import MultimodalSignDetector, HOSPITAL_SIGN_TAXONOMY


def test_sign_detector():
    print("Initializing MultimodalSignDetector...")
    detector = MultimodalSignDetector(use_gpu=True)

    # 1. Test Contextual RoI Generation
    fake_detections = [
        {"class_name": "door", "bbox_xyxy": [100.0, 200.0, 300.0, 600.0]},
        {"class_name": "fire_extinguisher", "bbox_xyxy": [450.0, 300.0, 550.0, 500.0]},
    ]
    rois = detector.generate_contextual_rois((720, 1280), fake_detections)
    print(f"Generated {len(rois)} contextual RoIs:")
    for r in rois:
        print(f"  RoI ({r[0]:.0f}, {r[1]:.0f}, {r[2]:.0f}, {r[3]:.0f}) hint={r[4]}")

    assert len(rois) >= 2, "Expected at least 2 contextual RoIs (door and extinguisher)"

    # 2. Test Synthetic Sign Classification
    # Create green "EMERGENCY EXIT" sign patch
    exit_patch = np.zeros((100, 300, 3), dtype=np.uint8)
    exit_patch[:, :] = (34, 197, 94)  # Green background
    cv2.putText(exit_patch, "EMERGENCY EXIT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cat, text, sym, conf, mod = detector.classify_crop(exit_patch, category_hint="exit_sign")
    print(f"Synthetic Exit Sign Result: cat={cat}, text='{text}', conf={conf:.2f}, mod={mod}")
    assert cat == "exit_sign", f"Expected exit_sign, got {cat}"

    # Create red "FIRE EXTINGUISHER" sign patch
    fire_patch = np.zeros((100, 300, 3), dtype=np.uint8)
    fire_patch[:, :] = (30, 30, 220)  # Red background
    cv2.putText(fire_patch, "FIRE EXTINGUISHER", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cat_f, text_f, sym_f, conf_f, mod_f = detector.classify_crop(fire_patch, category_hint="sign_fire_extinguisher")
    print(f"Synthetic Fire Extinguisher Sign Result: cat={cat_f}, text='{text_f}', conf={conf_f:.2f}, mod={mod_f}")
    assert cat_f == "sign_fire_extinguisher", f"Expected sign_fire_extinguisher, got {cat_f}"

    # Create icon-only biohazard symbol patch (yellow background, black triangles, no text)
    bio_patch = np.zeros((150, 150, 3), dtype=np.uint8)
    bio_patch[:, :] = (15, 165, 245)  # Amber/Yellow
    # Draw triangular warning shape
    pts = np.array([[75, 20], [20, 130], [130, 130]], np.int32)
    cv2.polylines(bio_patch, [pts], isClosed=True, color=(0, 0, 0), thickness=6)
    cat_b, text_b, sym_b, conf_b, mod_b = detector.classify_crop(bio_patch, category_hint="hazmat_sign")
    print(f"Synthetic Triangle Hazard Icon Result: cat={cat_b}, text='{text_b}', sym='{sym_b}', conf={conf_b:.2f}, mod={mod_b}")
    assert cat_b in ("hazmat_sign", "sign_radiation", "sign_caution_wet_floor"), f"Expected hazard category, got {cat_b}"

    # 3. Test 3D Spatial Deduplication
    pos1 = np.array([1.20, 2.50, 2.10])
    pos2 = np.array([1.25, 2.48, 2.12])  # 6cm away
    pos3 = np.array([3.50, 4.00, 2.10])  # 2.8m away

    id1, is_new1 = detector.deduplicate_sign_3d("exit_sign", pos1)
    id2, is_new2 = detector.deduplicate_sign_3d("exit_sign", pos2)
    id3, is_new3 = detector.deduplicate_sign_3d("exit_sign", pos3)

    print(f"Dedup check: pos1->ID={id1} (new={is_new1}), pos2->ID={id2} (new={is_new2}), pos3->ID={id3} (new={is_new3})")
    assert is_new1 is True
    assert is_new2 is False and id2 == id1, "pos2 should match existing landmark pos1"
    assert is_new3 is True and id3 != id1, "pos3 should be a new distinct landmark"

    # 4. Test detect_signs_in_frame with full frame, depth, and pose
    frame_h, frame_w = 720, 1280
    dummy_rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    dummy_depth = np.full((frame_h, frame_w), 2500, dtype=np.uint16)  # 2.5m depth

    # Stamp an exit sign above a door
    # Door is at [400, 250, 700, 700]
    # Exit sign is at [450, 120, 650, 200]
    dummy_rgb[120:200, 450:650] = (34, 197, 94)
    cv2.putText(dummy_rgb[120:200, 450:650], "EXIT", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    mock_detections = [
        {"class_name": "door", "bbox_xyxy": [400.0, 250.0, 700.0, 700.0], "confidence": 0.88}
    ]

    class MockIntrinsics:
        fx = 600.0
        fy = 600.0
        cx = 640.0
        cy = 360.0

    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 0.0  # camera origin

    detected_signs = detector.detect_signs_in_frame(
        frame_index=1,
        rgb_bgr=dummy_rgb,
        detections=mock_detections,
        depth_mm=dummy_depth,
        intrinsics=MockIntrinsics(),
        pose_matrix=pose,
        cadence=20,
    )

    print(f"Frame Detection Result: found {len(detected_signs)} signs:")
    for ds in detected_signs:
        print(f"  Sign: {ds.display_name} ({ds.canonical_name}) at bbox={ds.bbox_xyxy} conf={ds.confidence:.2f} mod={ds.modality}")

    assert len(detected_signs) >= 1, "Expected at least 1 detected sign above the door"
    assert detected_signs[0].canonical_name == "exit_sign"

    print("\nALL MULTIMODAL SIGN DETECTOR UNIT TESTS PASSED!")


if __name__ == "__main__":
    test_sign_detector()
