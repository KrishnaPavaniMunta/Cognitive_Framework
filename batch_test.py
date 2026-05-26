"""
batch_test.py
─────────────
Paste your URLs and targets below, then run:
    python batch_test.py

Results are appended to outputs/v3_inference_log.xlsx
Annotated images go to outputs/inference_output/
"""

import os, sys, tempfile, time, warnings
warnings.filterwarnings("ignore")

from infer_v3 import (
    MODEL_PATH, EXCEL_PATH, OUTPUT_DIR,
    get_or_create_workbook, run_image_inference,
    compute_result_image, save_annotated, write_row,
    detect_media_type,
)
from ultralytics import YOLO
import requests

# ── EDIT THIS LIST ─────────────────────────────────────────────────────────────
# Each entry: ("URL", "expected_class_or_[None]")
TESTS = [
    ("URL_1_HERE", "fire_extinguisher"),
    ("URL_2_HERE", "fire_extinguisher"),
    ("URL_3_HERE", "fire_extinguisher"),
    ("URL_4_HERE", "fire_extinguisher"),
    ("URL_5_HERE", "fire_extinguisher"),
    ("URL_6_HERE", "fire_extinguisher"),
    ("URL_7_HERE", "fire_extinguisher"),
    ("URL_8_HERE", "fire_extinguisher"),
    ("URL_9_HERE", "fire_extinguisher"),
    ("URL_10_HERE", "fire_extinguisher"),
]
# ──────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "Chrome/124.0 Safari/537.36"
}
DELAY = 1.5   # seconds between downloads

def main():
    print(f"\n{'='*62}")
    print("  Hospital V3 — Batch Inference")
    print(f"  {len(TESTS)} tests queued")
    print(f"{'='*62}\n")

    print("Loading model ...")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded.\n")

    wb, ws, col_map = get_or_create_workbook()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for i, (url, target) in enumerate(TESTS, 1):
        if url.endswith("_HERE"):
            print(f"[{i:02d}/{len(TESTS)}] SKIPPED — placeholder not filled")
            summary.append((i, url[:40], target, "SKIPPED", "-"))
            continue

        fname = url.split("?")[0].split("/")[-1][:45]
        print(f"[{i:02d}/{len(TESTS)}] {fname}")

        # Download with retry
        downloaded = False
        tmp_path   = None
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, stream=True, timeout=60,
                                 allow_redirects=True)
                r.raise_for_status()
                suffix = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                for chunk in r.iter_content(512 * 1024):
                    tmp.write(chunk)
                tmp.close()
                size_kb = os.path.getsize(tmp.name) / 1024
                print(f"         {size_kb:.0f} KB (attempt {attempt+1})")
                tmp_path   = tmp.name
                downloaded = True
                break
            except Exception as e:
                print(f"         attempt {attempt+1} failed: {e}")
                time.sleep(5)

        if not downloaded:
            summary.append((i, fname, target, "DL_ERROR", "-"))
            continue

        try:
            media_type = detect_media_type(url)
            detections = run_image_inference(model, tmp_path)
            detected_str, conf_str, result_type, notes = \
                compute_result_image(target, detections)
            save_annotated(model, tmp_path, url, media_type)

            next_row = ws.max_row + 1
            write_row(ws, col_map, next_row,
                      url, target, detected_str, conf_str,
                      result_type, notes, detections, media_type)
            wb.save(EXCEL_PATH)

            top = sorted(detections.items(),
                         key=lambda x: max(x[1]), reverse=True)[:4]
            top_str = " | ".join(f"{k}={max(v):.2f}" for k, v in top)
            print(f"         {result_type:<22}  target_conf={conf_str}")
            if top_str:
                print(f"         top detections: {top_str}")
            summary.append((i, fname, target, result_type, conf_str))

        except Exception as e:
            print(f"         INFERENCE ERROR: {e}")
            import traceback; traceback.print_exc()
            summary.append((i, fname, target, "INF_ERROR", "-"))
        finally:
            try: os.unlink(tmp_path)
            except: pass

        time.sleep(DELAY)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'#':>2}  {'File':<38}  {'Target':<20}  {'Result':<22}  Conf")
    print("-"*80)
    for idx, fn, tgt, rt, conf in summary:
        print(f"{idx:>2}  {fn:<38}  {tgt:<20}  {rt:<22}  {conf}")

    valid  = [r for r in summary if r[3] not in ("SKIPPED", "DL_ERROR", "INF_ERROR")]
    tp     = sum(1 for r in valid if r[3] == "TP")
    tp_lc  = sum(1 for r in valid if "Low" in r[3])
    fn_cnt = sum(1 for r in valid if r[3].startswith("FN"))
    fp_cnt = sum(1 for r in valid if r[3].startswith("FP"))
    tn_cnt = sum(1 for r in valid if r[3].startswith("TN"))
    errs   = len(summary) - len(valid)
    det    = (tp + tp_lc) / len(valid) * 100 if valid else 0

    print(f"\n  TP={tp}  TP(Low)={tp_lc}  FN={fn_cnt}  FP={fp_cnt}  "
          f"TN={tn_cnt}  Errors/Skip={errs}")
    print(f"  Detection rate: {det:.0f}%  (out of {len(valid)} valid tests)")
    print(f"\n  Excel  → {EXCEL_PATH}")
    print(f"  Images → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
