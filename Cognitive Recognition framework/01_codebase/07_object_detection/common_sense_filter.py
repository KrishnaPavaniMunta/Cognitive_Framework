# common_sense_filter.py


def apply_common_sense_rules(predictions, frame_height, frame_width):
    """
    2D and geometric common-sense rule pre-filtering for YOLO and Grounding DINO detections.
    """
    smart_predictions = []

    for x1, y1, x2, y2, conf, name in predictions:
        object_name = name.lower()
        box_width = max(1.0, float(x2) - float(x1))
        box_height = max(1.0, float(y2) - float(y1))
        y_mid = (float(y1) + float(y2)) * 0.5

        # RULE 1: Security cameras cannot be on or near the floor!
        if "camera" in object_name:
            if y_mid > (frame_height * 0.40):
                print(f"  [SPATIAL COMMON SENSE] Ignored Security Camera: Low in frame (y_mid={y_mid:.0f} > {frame_height*0.40:.0f}), not on ceiling/high wall!")
                continue

        # RULE 2: Hazmat signs should have realistic location/size/shape.
        if "hazmat" in object_name:
            if y_mid > (frame_height * 0.70):
                print("  [SPATIAL COMMON SENSE] Ignored Hazmat sign: Located on the floor!")
                continue

            if box_width > (frame_width * 0.45):
                print("  [SPATIAL COMMON SENSE] Ignored Hazmat sign: Way too huge for standard signage!")
                continue

            aspect_ratio = box_width / box_height
            if aspect_ratio < 0.65 or aspect_ratio > 1.45:
                print(f"  [SPATIAL COMMON SENSE] Ignored Hazmat sign: Aspect ratio ({aspect_ratio:.2f}) doesn't match standard square/diamond signage!")
                continue

        # RULE 3: Spillage should sit on the floor, not float mid-scene.
        if "spillage" in object_name or object_name == "spill":
            if float(y2) < (frame_height * 0.55):
                print("  [SPATIAL COMMON SENSE] Ignored Spillage: Not low enough in frame to be on floor!")
                continue

        # RULE 4: Radiator
        if "radiator" in object_name:
            if float(y2) < (frame_height * 0.20):
                print("  [SPATIAL COMMON SENSE] Ignored Radiator: On ceiling level!")
                continue
            aspect_ratio = box_width / box_height
            if aspect_ratio < 0.25 or aspect_ratio > 8.0:
                print(f"  [SPATIAL COMMON SENSE] Ignored Radiator: Aspect ratio ({aspect_ratio:.2f}) is too extreme!")
                continue
            if box_width > (frame_width * 0.85) and box_height > (frame_height * 0.85):
                print("  [SPATIAL COMMON SENSE] Ignored Radiator: Screen-filling background artifact!")
                continue

        smart_predictions.append((x1, y1, x2, y2, conf, name))

    return smart_predictions
