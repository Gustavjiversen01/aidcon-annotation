# AIDCON Dataset Class Configuration
# Use these EXACT class names when labeling in your annotation tool

AIDCON_CLASSES = {
    "dump_truck": 1,
    "excavator": 2,
    "backhoe_loader": 3,
    "wheel_loader": 4,
    "compactor": 5,
    "dozer": 6,
    "grader": 7,
    "car": 8,
    "other": 9
}

# Reverse mapping for visualization
CLASS_NAMES = {v: k for k, v in AIDCON_CLASSES.items()}

# Colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    1: (0, 0, 255),      # dump_truck - Red
    2: (0, 165, 255),    # excavator - Orange
    3: (0, 255, 255),    # backhoe_loader - Yellow
    4: (0, 255, 0),      # wheel_loader - Green
    5: (255, 255, 0),    # compactor - Cyan
    6: (255, 0, 0),      # dozer - Blue
    7: (255, 0, 255),    # grader - Magenta
    8: (128, 128, 128),  # car - Gray
    9: (255, 255, 255),  # other - White
}
