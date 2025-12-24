"""
Auto-annotate construction machinery using Grounding DINO + SAM-2.

This is an alternative implementation that directly uses:
- Grounding DINO for text-based object detection
- SAM-2 for high-quality segmentation

Requirements:
    pip install groundingdino-py torch torchvision supervision opencv-python

Usage:
    python auto_annotate_v2.py
"""

import os
import sys
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from collections import Counter

# Try to import required modules
try:
    import supervision as sv
except ImportError:
    print("Missing supervision. Run: pip install supervision")
    sys.exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUT_DIR = SCRIPT_DIR / "auto_annotations"
PREVIEW_DIR = SCRIPT_DIR / "auto_preview"

# AIDCON text prompts and class mapping
TEXT_PROMPTS = [
    "dump truck",
    "excavator",
    "backhoe loader",
    "wheel loader",
    "compactor",
    "road roller",
    "bulldozer",
    "grader",
    "motor grader",
    "car",
    "pickup truck",
    "van",
    "crane",
    "concrete mixer truck",
]

# Map detected labels to AIDCON class names
LABEL_TO_CLASS = {
    "dump truck": "dump_truck",
    "excavator": "excavator",
    "backhoe loader": "backhoe_loader",
    "backhoe": "backhoe_loader",
    "wheel loader": "wheel_loader",
    "front loader": "wheel_loader",
    "loader": "wheel_loader",
    "compactor": "compactor",
    "road roller": "compactor",
    "roller": "compactor",
    "bulldozer": "dozer",
    "dozer": "dozer",
    "grader": "grader",
    "motor grader": "grader",
    "car": "car",
    "pickup truck": "car",
    "pickup": "car",
    "van": "car",
    "truck": "dump_truck",
    "crane": "other",
    "concrete mixer truck": "other",
    "concrete mixer": "other",
    "mixer": "other",
}


def create_preview(image: np.ndarray, detections: sv.Detections, class_names: list) -> np.ndarray:
    """Create annotated preview image."""
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    annotated = image.copy()
    if detections.mask is not None:
        annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)

    if len(detections) > 0:
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
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) < 3:
                    continue

                points = approx.squeeze().tolist()
                if len(points) < 3:
                    continue

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


def run_with_florence2_sam2():
    """Use Florence-2 for detection and SAM-2 for segmentation."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Florence-2 model
    print("Loading Florence-2 model...")
    florence_model_id = "microsoft/Florence-2-large"
    florence_model = AutoModelForCausalLM.from_pretrained(
        florence_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)

    # Load SAM-2 model
    print("Loading SAM-2 model...")
    sam2_checkpoint = "facebook/sam2-hiera-large"
    sam2_model = build_sam2(sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return florence_model, florence_processor, sam2_predictor, device


def run_with_yolo_world_sam2():
    """Use YOLO-World for open-vocabulary detection and SAM-2 for segmentation."""
    from ultralytics import YOLO
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load YOLO-World model
    print("Loading YOLO-World model...")
    yolo_model = YOLO("yolov8x-worldv2.pt")
    yolo_model.set_classes(TEXT_PROMPTS)

    # Load SAM-2 model
    print("Loading SAM-2 model...")
    sam2_checkpoint = "facebook/sam2-hiera-large"
    sam2_model = build_sam2(sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return yolo_model, sam2_predictor, device


def run_with_owlv2_sam2():
    """Use OWLv2 for open-vocabulary detection and SAM-2 for segmentation."""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load OWLv2 model
    print("Loading OWLv2 model...")
    owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)

    # Load SAM-2 model
    print("Loading SAM-2 model...")
    sam2_checkpoint = "facebook/sam2-hiera-large"
    sam2_model = build_sam2(sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return owl_processor, owl_model, sam2_predictor, device


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(exist_ok=True)

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Try to use OWLv2 + SAM-2 (most reliable for open-vocabulary detection)
    print("\nLoading models...")
    print("=" * 50)

    try:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from PIL import Image

        print("Loading OWLv2 for open-vocabulary detection...")
        owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        owl_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(device)
        owl_model.eval()

        print("Loading SAM-2 for segmentation...")
        # SAM2 needs config and checkpoint
        sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        if device == "cuda":
            sam2_predictor.model = sam2_predictor.model.cuda()

        print("Models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("\nTrying alternative approach with just SAM-2 for mask generation...")
        # Fall back to simpler approach
        raise

    # Find images
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    # Filter out zip files that might have been extracted
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]

    if not image_files:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"\nFound {len(image_files)} images to process")
    print("-" * 50)

    # Prepare text queries for OWLv2
    text_queries = [[f"a photo of a {p}" for p in TEXT_PROMPTS]]

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Error loading image")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run OWLv2 detection
        print("  Running object detection...")
        inputs = owl_processor(text=text_queries, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owl_model(**inputs)

        # Post-process detections
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = owl_processor.post_process_object_detection(
            outputs,
            threshold=0.15,  # Lower threshold to catch more objects
            target_sizes=target_sizes
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        print(f"  Detected {len(boxes)} objects")

        if len(boxes) == 0:
            # Save empty annotation
            json_data = {
                "version": "5.0.0",
                "flags": {},
                "shapes": [],
                "imagePath": img_path.name,
                "imageHeight": image.shape[0],
                "imageWidth": image.shape[1]
            }
            json_path = OUTPUT_DIR / (img_path.stem + ".json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"  Saved: {json_path.name} (no detections)")
            continue

        # Run SAM-2 segmentation for each detection
        print("  Running segmentation...")
        sam2_predictor.set_image(image_rgb)

        all_masks = []
        class_names = []
        valid_boxes = []
        valid_scores = []

        for i, (box, score, label_idx) in enumerate(zip(boxes, scores, labels)):
            # Get the text prompt that matched
            prompt_text = TEXT_PROMPTS[label_idx]
            class_name = LABEL_TO_CLASS.get(prompt_text, "other")

            # Get SAM-2 mask using box prompt
            masks, mask_scores, _ = sam2_predictor.predict(
                box=box,
                multimask_output=True
            )

            # Take the best mask
            best_mask_idx = np.argmax(mask_scores)
            mask = masks[best_mask_idx]

            all_masks.append(mask)
            class_names.append(class_name)
            valid_boxes.append(box)
            valid_scores.append(score)

        # Create supervision Detections object
        # Ensure masks are boolean for supervision compatibility
        masks_array = np.array(all_masks, dtype=bool)
        detections = sv.Detections(
            xyxy=np.array(valid_boxes),
            mask=masks_array,
            confidence=np.array(valid_scores),
            class_id=np.arange(len(valid_boxes))
        )

        # Report counts
        counts = Counter(class_names)
        for cls, count in sorted(counts.items()):
            print(f"    - {cls}: {count}")

        # Save JSON annotation
        json_data = detections_to_labelme_json(
            img_path, detections, class_names, image.shape
        )
        json_path = OUTPUT_DIR / (img_path.stem + ".json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  Saved: {json_path.name}")

        # Save preview
        preview = create_preview(image, detections, class_names)
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
