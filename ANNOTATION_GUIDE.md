# Step-by-Step Annotation Guide

## Quick Answer: Pixel-Level vs Bounding Boxes

**Use POLYGON annotations (pixel-level)** - here's why:

| Model | Needs | Why |
|-------|-------|-----|
| U-Net | Pixel masks | Semantic segmentation - predicts class for every pixel |
| DINOv2 | Pixel masks | Same - needs to know exact shape |
| YOLO | Bounding boxes | But can be derived from polygons automatically |

**Good news:** You don't need to be pixel-perfect!
- 80-90% accuracy is fine for qualitative evaluation
- The AI-assist in AnyLabeling makes this much easier
- You're annotating ~3 images, not 1000

---

## Detailed Workflow

### Step 1: Launch AnyLabeling

Open a terminal/command prompt and run:

```bash
anylabeling
```

If that doesn't work, try:
```bash
python -m anylabeling.app
```

### Step 2: Open Your Images Folder

1. Click **File → Open Dir** (or Ctrl+U)
2. Navigate to: `C:\Users\gusta\OneDrive\Billeder\annotation_project\images`
3. Click "Select Folder"
4. You should see your 3 images listed in the left panel

### Step 3: Set Up Output Folder

1. Click **File → Change Output Dir**
2. Navigate to: `C:\Users\gusta\OneDrive\Billeder\annotation_project\labels_json`
3. Click "Select Folder"

### Step 4: Create Your Label List

Before annotating, set up the class names:

1. Click **File → Save Labels** or look for a labels.txt option
2. Or simply: when you draw your first polygon, type the class name exactly:
   - `dump_truck`
   - `excavator`
   - `backhoe_loader`
   - `wheel_loader`
   - `compactor`
   - `dozer`
   - `grader`
   - `car`
   - `other`

**Tip:** After typing a class name once, it appears in a dropdown for future annotations.

### Step 5: Enable AI-Assisted Annotation (Optional but Recommended)

This is the killer feature that saves hours:

1. Look for the **AI/Brain icon** in the toolbar (or View → AI Tools)
2. Select a model:
   - **Segment Anything (SAM)** - Best quality, slower
   - **YOLOv8-seg** - Faster, good enough
3. First time: It will download the model (~400MB for SAM)
4. Wait for "Model loaded" message

### Step 6: Annotate an Image

#### Method A: With AI Assist (Recommended)

1. Click on the image to select it
2. Press `A` or click the **Auto Segment** button
3. **Click inside** the vehicle you want to annotate
4. The AI will automatically create a polygon around it
5. If it's not perfect, you can:
   - Click again to add more area
   - Right-click to remove area
   - Press Enter to confirm
6. Type the class name (e.g., `excavator`) and press Enter
7. Repeat for all vehicles in the image

#### Method B: Manual Polygon (Without AI)

1. Press `P` or click the **Polygon** tool
2. Click around the edge of the vehicle to create points
3. Double-click or press Enter to close the polygon
4. Type the class name and press Enter

### Step 7: Save Your Work

- Press `Ctrl+S` after each image
- Or enable **File → Auto Save** to save automatically
- Check that JSON files appear in your `labels_json` folder

### Step 8: Move to Next Image

- Press `D` to go to next image
- Press `A` to go to previous image
- Or click on the image name in the left panel

### Step 9: Repeat for All Images

Annotate all 3 images:
- CompImg1.jpg
- CompImg2.jpg
- CompImg3.jpg

---

## How Accurate Should Annotations Be?

### Acceptable (80-90% edge accuracy):
```
    ┌─────────────┐
    │  ┌───────┐  │   The polygon roughly follows
    │  │ TRUCK │  │   the vehicle outline. Small
    │  └───────┘  │   gaps or overlaps are OK.
    └─────────────┘
```

### NOT needed (pixel-perfect):
```
You do NOT need to trace every tiny detail,
mirror, wheel spoke, or shadow edge.
```

### What matters:
1. **Capture the main body** of the vehicle
2. **Don't cut off major parts** (don't miss half the truck bed)
3. **Don't include too much background** (don't draw a huge box)
4. **Correct class label** - This is most important!

### Edge cases:

**Partially visible vehicles (at image edge):**
- Annotate only the visible part

**Overlapping vehicles:**
- Draw separate polygons for each
- It's OK if polygons overlap slightly

**Shadows:**
- Generally exclude shadows
- Unless the shadow makes the vehicle shape clearer

**Unsure what type:**
- Use `other` if you can't identify the machine type
- Or skip very unclear/tiny vehicles

---

## Keyboard Shortcuts (AnyLabeling)

| Key | Action |
|-----|--------|
| D | Next image |
| A | Previous image |
| P | Polygon tool |
| R | Rectangle tool (for bounding boxes) |
| Ctrl+S | Save |
| Ctrl+Z | Undo |
| Delete | Delete selected annotation |
| +/- | Zoom in/out |
| Space+Drag | Pan image |

---

## After Annotation: Convert to Masks

Once all 3 images are annotated:

```bash
cd C:\Users\gusta\OneDrive\Billeder\annotation_project
python convert_labels_to_masks.py
```

This will:
1. Read JSONs from `labels_json/`
2. Create PNG masks in `masks/`
3. Create visual previews in `masks_preview/`

Check the previews to verify your annotations look correct!

---

## Troubleshooting

### "No JSON files found"
- Make sure you saved (Ctrl+S) after annotating
- Check that Output Dir is set to `labels_json` folder

### AI model won't load
- Try a smaller model (YOLOv8 instead of SAM)
- Or just use manual polygon mode - it's fine for 3 images

### Polygon tool is hard to use
- Zoom in (scroll wheel or +/-) for precision
- Use fewer points - 10-15 points is usually enough
- You can edit points after: click polygon, then drag points

### Class name typo
- Click on the annotation in the list
- Right-click → Edit Label
- Or delete and redraw

---

## Example: What to Annotate in Your Images

Looking at your CompImg1.jpg, I can see:
- Several vehicles on the road (likely `car`)
- Construction machinery in the dirt area (look for `excavator`, `dump_truck`, `dozer`)

For CompImg2.jpg and CompImg3.jpg:
- Similar road construction scene
- Focus on the machinery in the excavated areas

**Start with the obvious, large vehicles first.** You can always add more annotations later.
