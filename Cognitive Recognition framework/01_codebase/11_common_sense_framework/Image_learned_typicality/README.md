# 🧠 OneThinker Object Typicality & Common-Sense Reasoning Extractor

This module uses **OneThinker** (`OneThink/OneThinker-8B`) to inspect images of any target object class (e.g., `security_camera`, `keyboard`, `exit_sign`, `hand_sanitizer`) and extract both:
1. **Machine-Readable Typicality & Spatial Grounding (`<class>_typicality.json`)**:
   - Target & anchor bounding boxes and points (`[x1, y1, x2, y2]` in both normalized `[0, 1]` and pixel coordinates).
   - Structured spatial triplets (`target --[predicate]--> anchor_surface`, e.g., `security_camera --[affixed_to]--> wall`).
   - Empirical **Typicality Scores** ($P(\text{predicate}, \text{anchor} \mid \text{class}) \in [0, 1]$) aggregated across all analyzed images of that class.
   - Co-occurring surrounding objects (`near ceiling_light`, `above door`, `near mouse`, etc.) and 2D vertical zone statistics.
2. **Human-Readable Interpretation (`<class>_interpretation.md` & `<class>_interpretation.txt`)**:
   - Executive synthesis of how the object typically looks (colors, shape, material), where it is typically found, what surface it attaches to or rests on, and what surrounds it.
   - Ranked typicality tables for anchor surfaces and surrounding objects.
   - Per-image natural-language interpretation and OneThinker's `<think>...</think>` chain-of-thought reasoning.

---

## 📁 Per-Class Output Directory Structure

All information for a given target class is stored in one self-contained folder under `04_outputs_runs_and_logs/outputs/common_sense_typicality/<target_class>/`:

```text
04_outputs_runs_and_logs/outputs/common_sense_typicality/
├── backpack/
│   ├── backpack_typicality.json       # Machine-readable structured data + typicality scores
│   ├── backpack_interpretation.md     # Human-readable Markdown report
│   ├── backpack_interpretation.txt    # Plain-text human-readable report
│   └── annotated_images/              # Images with target & anchor boxes drawn
└── security_camera/
    ├── security_camera_typicality.json
    ├── security_camera_interpretation.md
    ├── security_camera_interpretation.txt
    └── annotated_images/
```

> **Note:** Running the script again with new images for the same `--target-class` automatically appends/merges the new observations into that class's folder and recomputes the aggregated typicality scores (pass `--overwrite` to start fresh).

---

## 🚀 Usage Commands

Run from the **project root**:
```powershell
cd "D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework"
```

### 1. Run on a Root Folder Containing Multiple Class Folders (Target One Class, e.g., 40 Images of `security_camera`)
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework\02_datasets\CS_framework\Exit Sign" `
  --target-class "exit_sign" `
  --max-images 40
```

### 2. Run on ALL Class Folders Inside a Multi-Class Dataset Folder
If you omit `--target-class` and pass a root folder whose subfolders are class names (`door`, `electrical_cabinet`, `exit_sign`, `hand_sanitizer`, `hazmat_sign`, `security_camera`, `wet_caution_sign`, etc.), it processes every class into its own output folder automatically:
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\datasets\Hospital_Updated\datasets" `
  --max-images 40
```

### 3. Run on a Specific Folder of Images for Any Custom Class
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "path\to\my_keyboard_images" `
  --target-class "keyboard" `
  --max-images 50
```

### 4. Run on a Single Image
```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\Testing Images\consultation_room.jpg" `
  --target-class "security_camera"
```
