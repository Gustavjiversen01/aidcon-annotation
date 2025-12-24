"""
Convert X-AnyLabeling / LabelMe JSON annotations to PNG segmentation masks.
Compatible with AIDCON dataset format for U-Net, YOLO, and DINOv2 training.

Usage:
    python convert_labels_to_masks.py

The script will:
1. Read all JSON files from labels_json/
2. Create corresponding PNG masks in masks/
3. Generate a visualization overlay in masks_preview/ for verification
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from class_config import AIDCON_CLASSES, CLASS_COLORS, CLASS_NAMES

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
LABELS_DIR = SCRIPT_DIR / "labels_json"
MASKS_DIR = SCRIPT_DIR / "masks"
PREVIEW_DIR = SCRIPT_DIR / "masks_preview"


def normalize_class_name(label: str) -> str:
    """
    Normalize class names to match AIDCON format.
    Handles variations like 'Dump Truck', 'dump-truck', 'DUMP_TRUCK', etc.
    """
    # Convert to lowercase and replace spaces/hyphens with underscores
    normalized = label.lower().strip().replace(" ", "_").replace("-", "_")

    # Handle common variations
    variations = {
        "dumptruck": "dump_truck",
        "dump": "dump_truck",
        "truck": "dump_truck",
        "backhoe": "backhoe_loader",
        "backhoeloader": "backhoe_loader",
        "wheelloader": "wheel_loader",
        "loader": "wheel_loader",
        "bulldozer": "dozer",
        "bull_dozer": "dozer",
        "roller": "compactor",
        "motor_grader": "grader",
        "motorgrader": "grader",
        "vehicle": "car",
        "automobile": "car",
        "pickup": "car",
        "unknown": "other",
        "misc": "other",
    }

    if normalized in variations:
        return variations[normalized]

    return normalized


def json_to_mask(json_path: Path, img_shape: tuple) -> np.ndarray:
    """
    Convert a JSON annotation file to a segmentation mask.

    Args:
        json_path: Path to the JSON annotation file
        img_shape: Shape of the original image (height, width, channels)

    Returns:
        numpy array mask with class IDs as pixel values
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create empty mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    # Get shapes/annotations
    shapes = data.get('shapes', [])

    for shape in shapes:
        label = shape.get('label', '')
        points = shape.get('points', [])
        shape_type = shape.get('shape_type', 'polygon')

        if not points or not label:
            continue

        # Normalize class name
        normalized_label = normalize_class_name(label)

        if normalized_label not in AIDCON_CLASSES:
            print(f"  Warning: Unknown class '{label}' (normalized: '{normalized_label}'), skipping")
            continue

        class_id = AIDCON_CLASSES[normalized_label]
        points_array = np.array(points, dtype=np.int32)

        if shape_type == 'polygon':
            cv2.fillPoly(mask, [points_array], class_id)
        elif shape_type == 'rectangle':
            # Rectangle: points are [[x1,y1], [x2,y2]]
            if len(points) >= 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)

    return mask


def create_preview(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a visualization overlay of the mask on the original image.
    """
    overlay = image.copy()

    for class_id, color in CLASS_COLORS.items():
        class_mask = (mask == class_id)
        if np.any(class_mask):
            overlay[class_mask] = color

    # Blend original and overlay
    preview = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Add legend
    y_offset = 30
    for class_id, name in CLASS_NAMES.items():
        if np.any(mask == class_id):
            color = CLASS_COLORS[class_id]
            cv2.rectangle(preview, (10, y_offset - 15), (30, y_offset), color, -1)
            cv2.putText(preview, f"{class_id}: {name}", (35, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(preview, f"{class_id}: {name}", (35, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25

    return preview


def main():
    # Create output directories
    MASKS_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(exist_ok=True)

    # Find all JSON files
    json_files = list(LABELS_DIR.glob("*.json"))

    if not json_files:
        print("No JSON files found in labels_json/")
        print("\nTo annotate your images:")
        print("1. Install X-AnyLabeling: pip install anylabeling")
        print("   Or download from: https://github.com/CVHub520/X-AnyLabeling/releases")
        print("2. Open the 'images' folder in X-AnyLabeling")
        print("3. Use these class names when labeling:")
        for name in AIDCON_CLASSES.keys():
            print(f"   - {name}")
        print("4. Save annotations to 'labels_json' folder")
        print("5. Re-run this script")
        return

    print(f"Found {len(json_files)} annotation file(s)")
    print("-" * 50)

    converted_count = 0

    for json_path in json_files:
        base_name = json_path.stem
        print(f"\nProcessing: {base_name}")

        # Find corresponding image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            candidate = IMAGES_DIR / (base_name + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  Error: No matching image found for {json_path.name}")
            continue

        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Error: Could not read image {img_path}")
            continue

        print(f"  Image size: {img.shape[1]}x{img.shape[0]}")

        # Convert to mask
        mask = json_to_mask(json_path, img.shape)

        # Count objects per class
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # Remove background

        if len(unique_classes) == 0:
            print("  Warning: No valid annotations found in this file")
            continue

        print(f"  Classes found: {[CLASS_NAMES.get(c, f'unknown_{c}') for c in unique_classes]}")

        # Save mask
        mask_path = MASKS_DIR / (base_name + ".png")
        cv2.imwrite(str(mask_path), mask)
        print(f"  Saved mask: {mask_path.name}")

        # Save preview
        preview = create_preview(img, mask)
        preview_path = PREVIEW_DIR / (base_name + "_preview.jpg")
        # Resize preview if too large
        max_dim = 2000
        h, w = preview.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Saved preview: {preview_path.name}")

        converted_count += 1

    print("\n" + "=" * 50)
    print(f"Conversion complete: {converted_count}/{len(json_files)} files processed")
    print(f"\nOutput locations:")
    print(f"  Masks: {MASKS_DIR}")
    print(f"  Previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
