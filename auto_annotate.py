"""
Auto-annotate construction machinery using Grounded SAM 2.

This uses text prompts to automatically detect and segment vehicles,
giving you a strong starting point that you can refine manually.

Requirements:
    pip install autodistill-grounded-sam-2 supervision opencv-python

Usage:
    python auto_annotate.py
"""

import os
import sys
from pathlib import Path

# Check dependencies before importing
def check_dependencies():
    missing = []
    try:
        import autodistill_grounded_sam_2
    except ImportError:
        missing.append("autodistill-grounded-sam-2")
    try:
        import supervision
    except ImportError:
        missing.append("supervision")
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    if missing:
        print("Missing dependencies. Please run:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
import supervision as sv
import cv2
import json
import numpy as np

# Paths
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUT_DIR = SCRIPT_DIR / "auto_annotations"
PREVIEW_DIR = SCRIPT_DIR / "auto_preview"

# AIDCON class prompts -> class names
# The key is what Grounded SAM looks for, value is the AIDCON class name
ONTOLOGY = {
    "dump truck": "dump_truck",
    "excavator": "excavator",
    "backhoe loader": "backhoe_loader",
    "wheel loader": "wheel_loader",
    "front loader": "wheel_loader",  # Alternative name
    "compactor": "compactor",
    "road roller": "compactor",  # Alternative name
    "bulldozer": "dozer",
    "dozer": "dozer",
    "grader": "grader",
    "motor grader": "grader",
    "car": "car",
    "pickup truck": "car",
    "van": "car",
    "crane": "other",
    "concrete mixer": "other",
}


def create_preview(image: np.ndarray, detections: sv.Detections, class_names: list) -> np.ndarray:
    """Create annotated preview image."""
    # Annotate with masks
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    annotated = image.copy()
    annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)

    # Create labels
    labels = [f"{class_names[i]} ({detections.confidence[i]:.2f})"
              for i in range(len(detections))]
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    return annotated


def detections_to_labelme_json(
    image_path: Path,
    detections: sv.Detections,
    class_names: list,
    img_shape: tuple
) -> dict:
    """Convert supervision Detections to LabelMe/AnyLabeling JSON format."""
    shapes = []

    if detections.mask is not None:
        for i, mask in enumerate(detections.mask):
            # Convert binary mask to polygon points
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Simplify contour to reduce points
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) < 3:
                    continue

                points = approx.squeeze().tolist()
                if len(points) < 3:
                    continue

                # Ensure points are list of [x, y] pairs
                if isinstance(points[0], (int, float)):
                    points = [points]

                shapes.append({
                    "label": class_names[i],
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

    return {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageHeight": img_shape[0],
        "imageWidth": img_shape[1]
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(exist_ok=True)

    # Initialize model with ontology
    print("Loading Grounded SAM 2 model...")
    print("(First run will download model weights ~2GB)")

    base_model = GroundedSAM2(
        ontology=CaptionOntology(ONTOLOGY)
    )

    # Find images
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))

    if not image_files:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"\nFound {len(image_files)} images to process")
    print("-" * 50)

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Error loading image")
            continue

        # Run prediction
        print("  Running detection and segmentation...")
        detections = base_model.predict(str(img_path))

        # Get class names for each detection
        class_names = []
        if detections.class_id is not None:
            class_list = list(ONTOLOGY.values())
            # Remove duplicates while preserving order
            seen = set()
            unique_classes = []
            for c in class_list:
                if c not in seen:
                    unique_classes.append(c)
                    seen.add(c)

            for class_id in detections.class_id:
                if class_id < len(unique_classes):
                    class_names.append(unique_classes[class_id])
                else:
                    class_names.append("other")

        print(f"  Found {len(detections)} objects")
        if len(detections) > 0:
            # Count by class
            from collections import Counter
            counts = Counter(class_names)
            for cls, count in counts.items():
                print(f"    - {cls}: {count}")

        # Save JSON annotation (LabelMe format - can be opened in AnyLabeling)
        json_data = detections_to_labelme_json(
            img_path, detections, class_names, image.shape
        )
        json_path = OUTPUT_DIR / (img_path.stem + ".json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  Saved: {json_path.name}")

        # Save preview
        preview = create_preview(image, detections, class_names)
        # Resize for reasonable file size
        max_dim = 2000
        h, w = preview.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview = cv2.resize(preview, None, fx=scale, fy=scale)

        preview_path = PREVIEW_DIR / (img_path.stem + "_auto.jpg")
        cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  Preview: {preview_path.name}")

    print("\n" + "=" * 50)
    print("Auto-annotation complete!")
    print(f"\nOutput files:")
    print(f"  JSON annotations: {OUTPUT_DIR}")
    print(f"  Preview images:   {PREVIEW_DIR}")
    print(f"\nNext steps:")
    print("  1. Check the preview images to see detection quality")
    print("  2. Copy JSON files to 'labels_json' folder")
    print("  3. Open in AnyLabeling to refine/fix annotations")
    print("  4. Run convert_labels_to_masks.py to create final masks")


if __name__ == "__main__":
    main()
