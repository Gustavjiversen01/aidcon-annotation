# AIDCON-Compatible Annotation Project

Manual annotation of aerial construction site images using AIDCON dataset categories.

## AIDCON Classes (9 categories)

Use these **exact names** when labeling:

| Class ID | Name | Description |
|----------|------|-------------|
| 1 | `dump_truck` | Dump trucks, tipper trucks |
| 2 | `excavator` | Excavators, diggers |
| 3 | `backhoe_loader` | Backhoe loaders (tractor with front bucket + rear excavator) |
| 4 | `wheel_loader` | Front-end loaders, wheel loaders |
| 5 | `compactor` | Rollers, compactors |
| 6 | `dozer` | Bulldozers |
| 7 | `grader` | Motor graders |
| 8 | `car` | Cars, pickup trucks, vans |
| 9 | `other` | Other machinery not fitting above categories |

## Quick Start

### 1. Install Dependencies

```bash
cd annotation_project
pip install -r requirements.txt
```

### 2. Install Annotation Tool

**Option A: X-AnyLabeling (Recommended - has SAM AI assistance)**
```bash
pip install anylabeling
anylabeling
```
Or download the standalone .exe from:
https://github.com/CVHub520/X-AnyLabeling/releases

**Option B: LabelMe (Simpler, no AI)**
```bash
pip install labelme
labelme
```

### 3. Annotate Images

1. Open the annotation tool
2. Open the `images/` folder
3. Draw polygons around each construction vehicle
4. Label with the exact class names from the table above
5. Save JSON files to `labels_json/` folder

**Tips for accurate annotation:**
- Draw tight polygons around the entire vehicle
- Include shadows only if they're part of the vehicle silhouette
- For partially occluded vehicles, annotate only visible parts
- Use `other` for equipment you can't identify

### 4. Convert to Masks

After annotating, run:

```bash
python convert_labels_to_masks.py
```

This creates:
- `masks/` - PNG segmentation masks (class ID as pixel value)
- `masks_preview/` - Visualization overlays for verification

### 5. Verify Results

```bash
python visualize_masks.py
```

## Output Format

The masks are single-channel PNG images where:
- Pixel value 0 = background
- Pixel value 1-9 = class ID (see table above)

This format is compatible with:
- U-Net semantic segmentation
- YOLO instance segmentation (with additional conversion)
- DINOv2 fine-tuning

## Folder Structure

```
annotation_project/
├── images/              # Your input images (CompImg1.jpg, etc.)
├── labels_json/         # JSON annotations from labeling tool
├── masks/               # Output PNG masks
├── masks_preview/       # Visualization overlays
├── class_config.py      # Class definitions
├── convert_labels_to_masks.py
├── visualize_masks.py
└── requirements.txt
```
