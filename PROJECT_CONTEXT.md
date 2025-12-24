# Project Context: ML for Construction-Site Drone Imagery Segmentation

## Essential Information

| Field | Value |
|-------|-------|
| **Student** | Gustav Jes Iversen |
| **Supervisor** | Christian Igel (DIKU) |
| **Co-supervisor** | Venkanna Babu Guthula (vegu@di.ku.dk) |
| **Company** | Kortomatic ApS / Dansk Drone Kompagni |
| **Company Contact** | Hans Hansen (hans@kortomatic.com) |
| **ECTS** | 15 |
| **Deadline** | 16 January 2026 |
| **Language** | English |

## Project Title

**Machine Learning for Semantic Segmentation of Construction-Site Drone Imagery Using Open-Source Datasets**

## Core Objectives

### Phase 1: Dataset Study
- Identify suitable open-source datasets for aerial/drone semantic segmentation
- Assess: label schema, spatial resolution, license, domain similarity
- Primary dataset: **AIDCON** (9 classes: dump_truck, excavator, backhoe_loader, wheel_loader, compactor, dozer, grader, car, other)
- Document datasets for Kortomatic's future reference

### Phase 2: Model Development & Comparison
Compare different segmentation model families:
1. **Convolutional encoder-decoder** (e.g., U-Net)
2. **Instance segmentation** (e.g., YOLO, Mask R-CNN)
3. **Transformer-based** (e.g., DINOv2, SegFormer)

Analyze: performance, robustness, practicality

## Key Constraints

- **No manual annotation of company data** within project scope
- Company images used for **qualitative inspection only**
- Focus on **open-source datasets** (AIDCON as primary)
- Evaluate **transferability** from public benchmarks to company data

## Deliverables

1. Overview of relevant open-source datasets
2. Implemented and trained segmentation models
3. Quantitative evaluation on open-source data
4. **Qualitative analysis** on Kortomatic imagery (the 3 CompImg files)
5. Recommendations for future operational use

## Learning Goals (Exam Focus)

1. Justify choice of open-source segmentation datasets for industrial scenario
2. Implement and train conceptually different segmentation models (including transformers)
3. Design and interpret evaluation experiments (quantitative + qualitative)
4. Reflect critically on transfer to company data and use cases

## Current Progress: Annotation Project

Manual annotations created for qualitative evaluation:
- `CompImg1.jpg`: 25 objects detected
- `CompImg2.jpg`: 25 objects detected
- `CompImg3.jpg`: 7 objects detected

Masks use AIDCON class IDs (1-9) for direct comparison with models trained on AIDCON.

## Repository

https://github.com/Gustavjiversen01/aidcon-annotation

## Session Focus Reminder

When working on this project, always consider:
1. Does this support the **learning goals**?
2. Is this **transferability analysis** (AIDCON → Kortomatic)?
3. Am I comparing **different model architectures**?
4. Can I **quantify** on AIDCON and **qualify** on company data?
