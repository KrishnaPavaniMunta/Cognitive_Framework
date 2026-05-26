"""
hospital_constants.py — Single source of truth for shared constants.

Import from here instead of duplicating in each script:
    from hospital_constants import STATIC_CLASS_NAMES
"""

from __future__ import annotations

# Objects that are spatially anchored (static / placed).
# Used to:
#   1. Filter which detections are written to the spatial memory DB.
#   2. Serve as anchor correspondences for world-frame coordinate estimation.
# Compare by class NAME (string) to avoid numeric ID mapping confusion.
STATIC_CLASS_NAMES: frozenset[str] = frozenset({
    # Hospital Heavy Equipment (parked)
    "hospital_bed", "infusion_pump", "iv_stand", "monitor_hosp",
    "patient_monitor", "surgical_light", "vending_machines", "wheelchair",
    "hospital_stretcher",
    # Hospital Infrastructure
    "cabinet", "bench_hosp", "door", "reception_counter", "radiator",
    "bathroom_labels", "fire_extinguisher", "security_camera", "exit_sign",
    # Surgical Tools / Small Medical Items
    "iv_bag", "test_tube", "surgical_scissor", "spillage",
    # General Furniture / Appliances
    "bench", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    # Small Placed Objects (office/household)
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "bag",
    # Street Infrastructure
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    # Food Items (placed)
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
})
