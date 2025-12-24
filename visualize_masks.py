"""
Visualize existing masks overlaid on images.
Useful for verifying annotations are correct before training.

Usage:
    python visualize_masks.py
"""

import cv2
import numpy as np
from pathlib import Path
from class_config import CLASS_COLORS, CLASS_NAMES

SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "images"
MASKS_DIR = SCRIPT_DIR / "masks"


def main():
    mask_files = list(MASKS_DIR.glob("*.png"))

    if not mask_files:
        print("No mask files found in masks/")
        return

    print(f"Found {len(mask_files)} mask file(s)")
    print("Press any key to go to next image, 'q' to quit\n")

    for mask_path in mask_files:
        base_name = mask_path.stem

        # Find image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = IMAGES_DIR / (base_name + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"No image found for {mask_path.name}")
            continue

        # Load
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # Create overlay
        overlay = img.copy()
        for class_id, color in CLASS_COLORS.items():
            overlay[mask == class_id] = color

        # Blend
        preview = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        # Resize for display
        max_dim = 1200
        h, w = preview.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview = cv2.resize(preview, None, fx=scale, fy=scale)

        # Add info
        unique = np.unique(mask)
        unique = unique[unique > 0]
        info = f"{base_name} | Classes: {[CLASS_NAMES.get(c, c) for c in unique]}"
        cv2.putText(preview, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Mask Visualization (press key for next, q to quit)", preview)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
