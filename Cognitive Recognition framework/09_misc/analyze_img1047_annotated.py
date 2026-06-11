from pathlib import Path
import cv2
import math

video_path = Path(r"D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\04_outputs_runs_and_logs\AD_Rules_Outputs\IMG_1047_annotated.mp4")
out_sheet = video_path.with_name("IMG_1047_annotated_contact_sheet.jpg")
out_frames_dir = video_path.with_name("IMG_1047_frames")
out_frames_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sample_count = 20
indices = sorted(set(int(round(i * (max(total - 1, 1) / max(sample_count - 1, 1)))) for i in range(sample_count)))
frames = []

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    sec = idx / fps
    stamp = f"f={idx} t={sec:.1f}s"
    cv2.putText(frame, stamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, stamp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    frames.append((idx, frame))
    cv2.imwrite(str(out_frames_dir / f"frame_{idx:04d}.jpg"), frame)

cap.release()

if not frames:
    raise RuntimeError("No frames sampled")

thumb_w = 360
thumb_h = int(round(thumb_w * (height / max(width, 1))))
cols = 5
rows = math.ceil(len(frames) / cols)

sheet = 255 * (cv2.UMat(rows * thumb_h, cols * thumb_w, cv2.CV_8UC3).get())

for i, (_, frame) in enumerate(frames):
    r = i // cols
    c = i % cols
    thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    y1 = r * thumb_h
    y2 = y1 + thumb_h
    x1 = c * thumb_w
    x2 = x1 + thumb_w
    sheet[y1:y2, x1:x2] = thumb

cv2.imwrite(str(out_sheet), sheet)
print(f"VIDEO: {video_path}")
print(f"FPS: {fps:.2f}")
print(f"TOTAL: {total}")
print(f"SHEET: {out_sheet}")
print(f"FRAMES_DIR: {out_frames_dir}")
