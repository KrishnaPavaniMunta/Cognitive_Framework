from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image

proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda")

img_path = "Testing Images/Test4.png"
pil = Image.open(img_path).convert("RGB")
W, H = pil.size
print(f"Image size: {W}x{H}")

TEXT_THR = 0.25  # matches DINO_TEXT_THR in inference

def exact_dino_query(pil_image, phrase, canonical, box_thr, text_thr=TEXT_THR):
    """Reproduce _dino_query exactly."""
    prompt = phrase + " ."
    inputs = proc(images=pil_image, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"],
        threshold=box_thr, text_threshold=text_thr,
        target_sizes=[(H, W)]
    )[0]
    print(f"\n[{canonical}]  prompt='{phrase[:70]}...'")
    print(f"  box_thr={box_thr}  text_thr={text_thr}")
    if len(res["boxes"]) == 0:
        print("  NO DETECTIONS after thresholds")
    for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
        x1,y1,x2,y2 = [round(float(v),1) for v in box]
        area_frac = (x2-x1)*(y2-y1)/(W*H)
        print(f"  score={float(score):.3f}  label='{label}'  area={area_frac:.4f}")
    # also show with text_thr=0 to reveal what's being filtered
    res2 = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"],
        threshold=0.20, text_threshold=0.0,
        target_sizes=[(H, W)]
    )[0]
    print(f"  [raw, box>=0.20, text>=0.0]:")
    for box, score, label in zip(res2["boxes"], res2["scores"], res2["text_labels"]):
        x1,y1,x2,y2 = [round(float(v),1) for v in box]
        area_frac = (x2-x1)*(y2-y1)/(W*H)
        print(f"    score={float(score):.3f}  label='{label}'  area={area_frac:.4f}")

exact_dino_query(pil,
    "surgical scissors. stainless steel. instrument tray.",
    "surgical_scissor", box_thr=0.42)

exact_dino_query(pil,
    "stainless steel medical tray. rectangular or oval silver tray used in clinical or hospital setting.",
    "medical_tray", box_thr=0.40)

exact_dino_query(pil,
    "infusion pump. intravenous IV pump with digital display mounted on pole.",
    "infusion_pump", box_thr=0.35)
