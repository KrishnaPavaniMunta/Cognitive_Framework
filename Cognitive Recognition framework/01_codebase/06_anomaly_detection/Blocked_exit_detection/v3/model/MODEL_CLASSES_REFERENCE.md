# Model Detectables and Classes Reference

This document catalogs all models stored in `v3/model/`, their class counts, and the full list of detectable categories.

---

## 1. Summary Comparison

| Model File | Total Classes | Doors / Exits | General Obstacles / Clutter | Specialized Facility / PPE |
| :--- | :---: | :---: | :---: | :---: |
| **`yolo_trained_v3.pt`** | **109** | `door`, `exit_sign` | All COCO objects + `bag` + `spillage` | Hospital equipment, beds, PPE, signs |
| **`yolo_trained_v1.pt`** | **106** | `door` | All COCO objects | Hospital equipment, beds, PPE, signs |
| **`yolo_trained_v2.pt`** | **12** | `door`, `exit_sign` | Limited | PPE, sanitizer, safety & hazmat signs |
| **`Door_Detection_trained.pt`** | **3** | `Open`, `Close`, `Semi` | None (Door state classifier) | None |

> **Key takeaway:** `yolo_trained_v3.pt` is a strict superset of `yolo_trained_v1.pt`. It contains all 106 classes from V1 plus 3 additions: `exit_sign`, `bag`, and `spillage`.

---

## 2. `yolo_trained_v3.pt` (109 Classes)

Unified model containing both doors, emergency exit signs, all hospital equipment, general everyday clutter, and floor hazards.

### Categorized Breakdown:
- **Exits & Architecture:**
  - `door`, `exit_sign`, `cabinet`, `reception_counter`
- **Floor Hazards & Clutter:**
  - `bag`, `spillage`, `backpack`, `handbag`, `suitcase`
- **Medical Furniture & Wheelchairs:**
  - `hospital_bed`, `hospital_stretcher`, `wheelchair`, `bench_hosp`, `bed`, `bench`, `chair`, `couch`
- **Medical Equipment & Devices:**
  - `infusion_pump`, `iv_bag`, `iv_stand`, `monitor_hosp`, `patient_monitor`, `radiator`, `surgical_light`, `surgical_scissor`, `test_tube`, `vending_machines`
- **People & Roles:**
  - `healthcare_worker`, `patient`, `person`
- **Safety, PPE & Facility Items:**
  - `fire_extinguisher`, `glove`, `hair_net`, `mask`, `nasal_cannula`, `security_camera`, `bathroom_labels`
- **Everyday Objects (COCO subset):**
  - `airplane`, `apple`, `banana`, `baseball bat`, `baseball glove`, `bear`, `bicycle`, `bird`, `boat`, `book`, `bottle`, `bowl`, `broccoli`, `bus`, `cake`, `car`, `carrot`, `cat`, `cell phone`, `clock`, `cow`, `cup`, `dining table`, `dog`, `donut`, `elephant`, `fire hydrant`, `fork`, `frisbee`, `giraffe`, `hair drier`, `horse`, `hot dog`, `keyboard`, `kite`, `knife`, `laptop`, `microwave`, `motorcycle`, `mouse`, `orange`, `oven`, `parking meter`, `pizza`, `potted plant`, `refrigerator`, `remote`, `sandwich`, `scissors`, `sheep`, `sink`, `skateboard`, `skis`, `snowboard`, `spoon`, `sports ball`, `stop sign`, `surfboard`, `teddy bear`, `tennis racket`, `tie`, `toaster`, `toilet`, `toothbrush`, `traffic light`, `train`, `truck`, `tv`, `umbrella`, `vase`, `wine glass`, `zebra`

### Full Alphabetical List (109):
1. `airplane`
2. `apple`
3. `backpack`
4. `bag` *(New in V3)*
5. `banana`
6. `baseball bat`
7. `baseball glove`
8. `bathroom_labels`
9. `bear`
10. `bed`
11. `bench`
12. `bench_hosp`
13. `bicycle`
14. `bird`
15. `boat`
16. `book`
17. `bottle`
18. `bowl`
19. `broccoli`
20. `bus`
21. `cabinet`
22. `cake`
23. `car`
24. `carrot`
25. `cat`
26. `cell phone`
27. `chair`
28. `clock`
29. `couch`
30. `cow`
31. `cup`
32. `dining table`
33. `dog`
34. `donut`
35. `door`
36. `elephant`
37. `exit_sign` *(New in V3)*
38. `fire hydrant`
39. `fire_extinguisher`
40. `fork`
41. `frisbee`
42. `giraffe`
43. `glove`
44. `hair drier`
45. `hair_net`
46. `handbag`
47. `healthcare_worker`
48. `horse`
49. `hospital_bed`
50. `hospital_stretcher`
51. `hot dog`
52. `infusion_pump`
53. `iv_bag`
54. `iv_stand`
55. `keyboard`
56. `kite`
57. `knife`
58. `laptop`
59. `mask`
60. `microwave`
61. `monitor_hosp`
62. `motorcycle`
63. `mouse`
64. `nasal_cannula`
65. `orange`
66. `oven`
67. `parking meter`
68. `patient`
69. `patient_monitor`
70. `person`
71. `pizza`
72. `potted plant`
73. `radiator`
74. `reception_counter`
75. `refrigerator`
76. `remote`
77. `sandwich`
78. `scissors`
79. `security_camera`
80. `sheep`
81. `sink`
82. `skateboard`
83. `skis`
84. `snowboard`
85. `spillage` *(New in V3)*
86. `spoon`
87. `sports ball`
88. `stop sign`
89. `suitcase`
90. `surfboard`
91. `surgical_light`
92. `surgical_scissor`
93. `teddy bear`
94. `tennis racket`
95. `test_tube`
96. `tie`
97. `toaster`
98. `toilet`
99. `toothbrush`
100. `traffic light`
101. `train`
102. `truck`
103. `tv`
104. `umbrella`
105. `vase`
106. `vending_machines`
107. `wheelchair`
108. `wine glass`
109. `zebra`

---

## 3. `yolo_trained_v1.pt` (106 Classes)

Same as V3, except it lacks `bag`, `exit_sign`, and `spillage`.

### Full Alphabetical List (106):
`airplane`, `apple`, `backpack`, `banana`, `baseball bat`, `baseball glove`, `bathroom_labels`, `bear`, `bed`, `bench`, `bench_hosp`, `bicycle`, `bird`, `boat`, `book`, `bottle`, `bowl`, `broccoli`, `bus`, `cabinet`, `cake`, `car`, `carrot`, `cat`, `cell phone`, `chair`, `clock`, `couch`, `cow`, `cup`, `dining table`, `dog`, `donut`, `door`, `elephant`, `fire hydrant`, `fire_extinguisher`, `fork`, `frisbee`, `giraffe`, `glove`, `hair drier`, `hair_net`, `handbag`, `healthcare_worker`, `horse`, `hospital_bed`, `hospital_stretcher`, `hot dog`, `infusion_pump`, `iv_bag`, `iv_stand`, `keyboard`, `kite`, `knife`, `laptop`, `mask`, `microwave`, `monitor_hosp`, `motorcycle`, `mouse`, `nasal_cannula`, `orange`, `oven`, `parking meter`, `patient`, `patient_monitor`, `person`, `pizza`, `potted plant`, `radiator`, `reception_counter`, `refrigerator`, `remote`, `sandwich`, `scissors`, `security_camera`, `sheep`, `sink`, `skateboard`, `skis`, `snowboard`, `spoon`, `sports ball`, `stop sign`, `suitcase`, `surfboard`, `surgical_light`, `surgical_scissor`, `teddy bear`, `tennis racket`, `test_tube`, `tie`, `toaster`, `toilet`, `toothbrush`, `traffic light`, `train`, `truck`, `tv`, `umbrella`, `vase`, `vending_machines`, `wheelchair`, `wine glass`, `zebra`

---

## 4. `yolo_trained_v2.pt` (12 Specialized Classes)

Focused strictly on facility compliance, PPE, and signage:

1. `bin`
2. `door`
3. `electrical_cabinet`
4. `exit_sign`
5. `glove`
6. `hair_net`
7. `hand_sanitizer`
8. `hazmat_sign`
9. `mask`
10. `security_camera`
11. `test_tube`
12. `wet_floor_sign`

---

## 5. `Door_Detection_trained.pt` (3 State Classes)

Door state classification model:
1. `Close`
2. `Open`
3. `Semi`
