"""
Convert LabelMe JSON annotations to COCO JSON format.
This ensures compatibility with the AIDCON dataset format.

AIDCON uses COCO format with these 9 categories:
    1: dump_truck
    2: excavator
    3: backhoe_loader
    4: wheel_loader
    5: compactor
    6: dozer
    7: grader
    8: car
    9: other

Usage:
    python convert_to_coco.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
LABELS_DIR = SCRIPT_DIR / "labels_json"
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUT_DIR = SCRIPT_DIR / "coco_format"

# AIDCON category mapping (matching their exact format)
AIDCON_CATEGORIES = [
    {"id": 1, "name": "dump_truck", "supercategory": "construction_machinery"},
    {"id": 2, "name": "excavator", "supercategory": "construction_machinery"},
    {"id": 3, "name": "backhoe_loader", "supercategory": "construction_machinery"},
    {"id": 4, "name": "wheel_loader", "supercategory": "construction_machinery"},
    {"id": 5, "name": "compactor", "supercategory": "construction_machinery"},
    {"id": 6, "name": "dozer", "supercategory": "construction_machinery"},
    {"id": 7, "name": "grader", "supercategory": "construction_machinery"},
    {"id": 8, "name": "car", "supercategory": "vehicle"},
    {"id": 9, "name": "other", "supercategory": "other"},
]

# Name to ID mapping
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in AIDCON_CATEGORIES}


def normalize_class_name(label: str) -> str:
    """Normalize class names to match AIDCON format."""
    normalized = label.lower().strip().replace(" ", "_").replace("-", "_")

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

    return variations.get(normalized, normalized)


def polygon_to_coco_segmentation(points):
    """Convert polygon points to COCO segmentation format (flat list)."""
    # COCO format: [x1, y1, x2, y2, x3, y3, ...]
    flat_points = []
    for point in points:
        flat_points.extend([float(point[0]), float(point[1])])
    return [flat_points]


def polygon_to_bbox(points):
    """Calculate bounding box from polygon points."""
    points_array = np.array(points)
    x_min = float(np.min(points_array[:, 0]))
    y_min = float(np.min(points_array[:, 1]))
    x_max = float(np.max(points_array[:, 0]))
    y_max = float(np.max(points_array[:, 1]))

    # COCO bbox format: [x, y, width, height]
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def polygon_area(points):
    """Calculate polygon area using shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def convert_labelme_to_coco():
    """Convert all LabelMe JSONs to a single COCO format JSON."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Kortomatic Company Images - AIDCON Compatible Annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Gustav Jes Iversen",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Research Use Only",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": AIDCON_CATEGORIES
    }

    # Find all LabelMe JSON files
    json_files = sorted(LABELS_DIR.glob("*.json"))

    if not json_files:
        print("No JSON files found in labels_json/")
        return

    print(f"Converting {len(json_files)} annotation files to COCO format...")
    print("-" * 50)

    annotation_id = 1

    for image_id, json_path in enumerate(json_files, start=1):
        print(f"\nProcessing: {json_path.name}")

        with open(json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)

        # Get image info
        image_filename = labelme_data.get("imagePath", json_path.stem + ".jpg")
        image_height = labelme_data.get("imageHeight", 0)
        image_width = labelme_data.get("imageWidth", 0)

        # If dimensions not in JSON, read from image
        if image_height == 0 or image_width == 0:
            img_path = IMAGES_DIR / image_filename
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    image_height, image_width = img.shape[:2]

        # Add image entry
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": image_width,
            "height": image_height,
            "license": 1,
            "date_captured": ""
        })

        print(f"  Image: {image_filename} ({image_width}x{image_height})")

        # Process shapes/annotations
        shapes = labelme_data.get("shapes", [])
        class_counts = {}

        for shape in shapes:
            label = shape.get("label", "")
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")

            if not points or not label:
                continue

            # Only handle polygons
            if shape_type != "polygon":
                print(f"  Warning: Skipping non-polygon shape type: {shape_type}")
                continue

            if len(points) < 3:
                print(f"  Warning: Skipping polygon with less than 3 points")
                continue

            # Normalize class name
            normalized_label = normalize_class_name(label)

            if normalized_label not in CATEGORY_NAME_TO_ID:
                print(f"  Warning: Unknown class '{label}' -> '{normalized_label}', using 'other'")
                normalized_label = "other"

            category_id = CATEGORY_NAME_TO_ID[normalized_label]

            # Count classes
            class_counts[normalized_label] = class_counts.get(normalized_label, 0) + 1

            # Create COCO annotation
            segmentation = polygon_to_coco_segmentation(points)
            bbox = polygon_to_bbox(points)
            area = polygon_area(points)

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

            annotation_id += 1

        # Print class summary
        for cls_name, count in sorted(class_counts.items()):
            print(f"    {cls_name}: {count}")

    # Save COCO JSON
    output_path = OUTPUT_DIR / "annotations.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)

    print("\n" + "=" * 50)
    print(f"COCO format conversion complete!")
    print(f"\nOutput: {output_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"\nCategories (AIDCON compatible):")
    for cat in AIDCON_CATEGORIES:
        count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == cat['id'])
        if count > 0:
            print(f"  {cat['id']}: {cat['name']} - {count} instances")


if __name__ == "__main__":
    convert_labelme_to_coco()
