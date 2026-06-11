import cv2
from pathlib import Path
import math
from PIL import Image, ImageDraw

video_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\AD_Rules_Outputs\Person_blocking_hospital_exit_202606091138_annotated_rerun3.mp4")
out_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\AD_Rules_Outputs\Person_blocking_hospital_exit_202606091138_rerun3_contact_sheet.jpg")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
sample_fracs = [0.00, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.96]
sample_indices = [min(frame_count - 1, max(0, int(fr * frame_count))) for fr in sample_fracs] if frame_count > 0 else [0]

frames = []
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append((idx, Image.fromarray(rgb)))
cap.release()

if not frames:
    raise RuntimeError("No frames extracted")

thumb_w = 410
cards = []
for idx, im in frames:
    ratio = thumb_w / im.width
    thumb = im.resize((thumb_w, max(1, int(im.height * ratio))))
    card = Image.new("RGB", (thumb_w, thumb.height + 34), "white")
    card.paste(thumb, (0, 0))
    d = ImageDraw.Draw(card)
    d.text((8, thumb.height + 8), f"frame {idx}/{frame_count}  t={idx / fps:.2f}s", fill="black")
    cards.append(card)

cols = 3
rows = math.ceil(len(cards) / cols)
max_w = max(c.width for c in cards)
max_h = max(c.height for c in cards)
sheet = Image.new("RGB", (cols * max_w + (cols - 1) * 12, rows * max_h + (rows - 1) * 12), "#dddddd")
for i, card in enumerate(cards):
    x = (i % cols) * (max_w + 12)
    y = (i // cols) * (max_h + 12)
    sheet.paste(card, (x, y))

sheet.save(out_path)
print({"video": str(video_path), "contact_sheet": str(out_path), "frames": frame_count, "fps": fps, "samples": sample_indices})
