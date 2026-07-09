"""Grounding DINO prompt and threshold configuration."""

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"

# Formatted as: "class_key": ("custom phrase detection prompt", individual_confidence_threshold)
DINO_FALLBACK = {
    "surgical_scissor": ("surgical scissors. stainless steel scissors. metal surgical scissors with pointed blades.", 0.45),
    "surgical_light": ("round surgical operating light. operating room light on articulated arm. large circular surgical light.", 0.40),
    "glove": ("blue surgical glove on hand. purple nitrile glove on hand. white latex medical glove on hand.", 0.45),
    "mask": ("blue surgical face mask. white medical face mask. blue medical face mask worn by person.", 0.35),
    "hair_net": ("surgical hair net on head. ble mesh hair cover on head. disposable bouffant cap.", 0.45),
    "radiator": ("radiator heater panel mounted on wall. ribbed slotted steel heating element.", 0.50),
    "exit_sign": ("green exit sign. illuminated green exit sign with arrow. green rectangular sign mounted above doorway.", 0.45),
    "door": ("hospital corridor door. fire door, emergency exit door.", 0.45),
    "medical_tray": ("steel silver medical tray. flat rectangular plastic medical tray. metal instrument tray.", 0.40),
    "hand_sanitizer": ("wall-mounted soap sanitizer. dispenser. handwash. sink wall dispenser", 0.40),
    "bin": ("waste bin. trash bin. yellow waste bin. yellow black striped waste bin.", 0.35),
    "hazmat_sign": ("hazard sign. caution. fire symbol. diamond shape. triangle shaped yellow warning. Danger.", 0.40),
    "utility_trolley": ("trolley with multiple shelves. stand with wheels. rolling cart with shelves and push handle. wheeled trolley.", 0.42),
    "oxygen_pump": ("upright oxygen concentrator machine. oxygen concentrator tower with front control panel, vents, wheels, and oxygen tubing. hospital oxygen concentrator unit plugged into wall.", 0.58),
    "power_socket": ("electrical switchboard. electrical panel. distribution board. breaker panel. fuse box. electrical cabinet.", 0.20),
    "iv_stand": ("hospital iv stand pole with wheeled base and hanging hooks. intravenous drip stand.", 0.48),
    "infusion_pump": ("infusion pump device mounted on iv pole with monitor and control buttons.", 0.50),
}

# Classes within a group must NOT share overlapping vocabulary in their prompts,
# otherwise label-matching after batched DINO inference becomes ambiguous.
# Pairs with genuine conceptual/vocabulary overlap are kept solo.
DINO_BATCH_GROUPS = [
    ["surgical_light", "radiator", "hand_sanitizer", "hazmat_sign", "switch_board"],  # distinct vocab, safe to batch
    ["glove", "mask", "hair_net"],  # PPE, distinct body locations, safe to batch
    ["bin"],  # standalone, no clean overlap partner left in this set
    ["door"],  # overlaps with exit_sign vocabulary - solo
    ["exit_sign"],  # overlaps with door vocabulary - solo
    ["surgical_scissor"],  # overlaps with medical_tray ("instrument tray") - solo
    ["medical_tray"],  # overlaps with surgical_scissor - solo
    ["utility_trolley"],  # overlaps with medical_tray/trolley-cart vocabulary - solo
    ["oxygen_pump"],  # overlaps with iv_stand/infusion contexts - solo
    ["iv_stand"],  # overlaps with infusion_pump ("pole"/"IV") - solo
    ["infusion_pump"],  # overlaps with iv_stand - solo
    
]
