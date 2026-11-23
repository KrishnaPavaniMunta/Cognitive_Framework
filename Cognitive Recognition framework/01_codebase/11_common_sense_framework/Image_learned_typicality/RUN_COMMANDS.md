# 🧠 OneThinker Object Typicality & Reasoning — Run Commands

All commands below are run from the **project root folder**:
```powershell
cd "D:\Object Detection Model\yolo_tr\yolo_tr\Cognitive Recognition framework"
```

---

## ⚡ 1. Run on One Specific Class from the Multi-Class Dataset Folder
*(Extracts 40 images of `security_camera` from `02_datasets\datasets\Hospital_Updated\datasets` and saves machine + human outputs in `04_outputs_runs_and_logs\outputs\common_sense_typicality\security_camera\`)*

```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\datasets\Hospital_Updated\datasets" `
  --target-class "security_camera" `
  --max-images 40
```

Other dataset classes you can pass to `--target-class`:
- `security_camera`
- `exit_sign`
- `electrical_cabinet`
- `door`
- `hand_sanitizer`
- `hazmat_sign`
- `wet_caution_sign`
- `glove`
- `hairnet`
- `mask`
- `tube`

---

## ⚡ 2. Run on ALL Class Folders Inside a Multi-Class Dataset Folder Automatically
*(Omitting `--target-class` automatically discovers every class subfolder and creates one output folder per class)*

```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\datasets\Hospital_Updated\datasets" `
  --max-images 40
```

---

## ⚡ 3. Run on Any Custom Folder of Images for a Target Class
*(Point `--input` to any folder containing images of your target object, e.g., `keyboard`, `fire_extinguisher`, `security_camera`)*

```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "path\to\your_image_folder" `
  --target-class "keyboard" `
  --max-images 50
```

---

## ⚡ 4. Run on a Single Image
*(Appends the observation into that class's folder and updates both the `.json` and human-readable `.md`/`.txt` reports)*

```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\Testing Images\consultation_room.jpg" `
  --target-class "security_camera"
```

---

## 🔄 5. Re-Run & Overwrite Existing Class Records (Start Fresh)
*(By default, the script resumes and merges new images into the existing class folder. Add `--overwrite` to recompute from scratch)*

```powershell
.\07_environment_and_project_meta\.venv-gpu311\Scripts\python.exe `
  01_codebase\11_common_sense_framework\Image_learned_typicality\extract_object_typicality.py `
  --input "02_datasets\datasets\Hospital_Updated\datasets" `
  --target-class "security_camera" `
  --max-images 40 `
  --overwrite
```

---

## 📁 Output Location Per Class

Every class is saved in its own dedicated directory under:
`04_outputs_runs_and_logs\outputs\common_sense_typicality\<target_class>\`

| File / Folder | Description |
| :--- | :--- |
| `<target_class>_typicality.json` | **Machine-readable** typicality scores, anchor surface probabilities, surrounding objects, appearance priors, and per-image bounding boxes/points. |
| `<target_class>_interpretation.md` | **Human-readable** Markdown report with executive common-sense synthesis, ranked tables, and per-image `<think>` reasoning. |
| `<target_class>_interpretation.txt` | **Human-readable** plain-text version of the report. |
| `annotated_images\` | Visual overlays showing the grounded target bounding box, anchor surface box, and interaction predicate on each image. |
