# common_sense_filter.py


def apply_common_sense_rules(predictions, frame_height, frame_width):
    """
    Teaches the robot basic real-world common sense so it stops making silly mistakes.
    You can keep adding new rules here as you find more false positives!
    """
    smart_predictions = []

    for x1, y1, x2, y2, conf, name in predictions:
        object_name = name.lower()
        box_width = max(1.0, float(x2) - float(x1))
        box_height = max(1.0, float(y2) - float(y1))

        # RULE 1: Security cameras cannot be on the floor!
        if "camera" in object_name:
            # If the top of the box (y1) is halfway down the screen or lower...
            if y1 > (frame_height / 2):
                print("  [COMMON SENSE] Ignored a fake Security Camera on the floor!")
                continue  # Skip it! Do not add it to our smart list.

        # RULE 2: Hazmat signs should have realistic location/size/shape.
        if "hazmat" in object_name:
            # Test 1: Is it on the floor? (Bottom 20% of the screen)
            if y1 > (frame_height * 0.8):
                print("  [COMMON SENSE] Ignored Hazmat sign: It's on the floor!")
                continue

            # Test 2: Is it unrealistically huge?
            if box_width > (frame_width * 0.5):
                print("  [COMMON SENSE] Ignored Hazmat sign: Way too huge!")
                continue

            # Test 3: Hazmat signs are usually near-square/diamond in projection.
            aspect_ratio = box_width / box_height
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                print(
                    f"  [COMMON SENSE] Ignored Hazmat sign: Aspect ratio ({aspect_ratio:.2f}) doesn't match a standard sign!"
                )
                continue

        # RULE 3: Spillage should sit on the floor, not float mid-scene.
        if "spillage" in object_name or object_name == "spill":
            # Use the bottom of the box as a cheap floor-contact proxy.
            if y2 < (frame_height * 0.75):
                print("  [COMMON SENSE] Ignored Spillage: Not low enough in the frame to be on the floor!")
                continue

        # ---------------------------------------------------------
        # RULE: RADIATOR
        # ---------------------------------------------------------
        if "radiator" in object_name:
            
            # Test 1: The Gravity Rule (Is it on the ceiling?)
            # y2 is the BOTTOM of the bounding box. 
            # If the bottom of the object is in the top 20% of the screen, it's a light/vent!
            if y2 < (frame_height * 0.2):
                print(f"  [COMMON SENSE] Ignored Radiator: It's on the ceiling (likely a vent/light)!")
                continue 
                
            # Test 2: The Shape Rule (Aspect Ratio)
            aspect_ratio = box_width / box_height
            
            # If it is super tall and skinny (ratio < 0.3) like a door frame...
            # OR if it is incredibly wide and flat (ratio > 8.0) like a floorboard...
            if aspect_ratio < 0.3 or aspect_ratio > 8.0:
                print(f"  [COMMON SENSE] Ignored Radiator: Shape is too extreme to be a radiator!")
                continue

            # Test 3: The Giant Rule
            # Radiators shouldn't take up the entire camera view unless the robot hit the wall.
            if box_width > (frame_width * 0.8) and box_height > (frame_height * 0.8):
                print(f"  [COMMON SENSE] Ignored Radiator: Taking up the whole screen, probably a textured wall!")
                continue
        # ---------------------------------------------------------
        # ADD NEW RULES HERE LATER!
        # Whenever you see the robot making a silly mistake,
        # write a new rule for it right here.
        # ---------------------------------------------------------

        # If the object passes the common sense tests, keep it!
        smart_predictions.append((x1, y1, x2, y2, conf, name))

    return smart_predictions
