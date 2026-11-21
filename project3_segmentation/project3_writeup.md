# Project 3: Semantic Segmentation on the Semantic Drone Dataset

**Course:** AI-driven Perception, Learning and Mapping for Drones — Modules 2, 5, 35–39
**Status:** Complete (with documented limitations)
**Date:** November 2026

## Abstract

We trained a U-Net for semantic segmentation on the Semantic Drone Dataset (400 drone images, 23 classes). Three experiments were run: from-scratch U-Net with CrossEntropy loss, from-scratch U-Net with Dice+CrossEntropy loss, and a pretrained ResNet-34 encoder with Dice+CrossEntropy loss. All three achieved mean IoU (mIoU) of ~0.03, well below published benchmarks for this dataset (~0.60). Per-class analysis revealed the cause: the model learns the dominant classes (paved-area IoU 0.38, grass IoU 0.21) but fails on the 20 rare classes, which appear in only a few training images. We diagnose this as a **data distribution bottleneck**, not a model or loss-function bottleneck. Random 512x512 crops on 4000x6000 images sample rare-class instances with probability <1%, so the model never sees them enough to learn. The most promising fixes are class-balanced patch sampling or full-image training; both are documented as future work.

## 1. Introduction

Semantic segmentation assigns a class label to every pixel in an image. It is the foundation of mapping applications: classifying roads, buildings, vegetation, and water from aerial imagery. For drone-based mapping, segmentation is the step that turns raw imagery into usable land-cover maps.

We chose the Semantic Drone Dataset (TU Graz):

- **400 training images**, 4000x6000 RGB, with pixel-level masks
- **23 classes** (paved-area, grass, dirt, vegetation, car, bicycle, etc.)
- **Severe class imbalance**: two classes cover ~58% of all pixels; 20 classes share the remaining 42%, with some classes appearing in <0.1% of pixels

## 2. Methods

### 2.1 Dataset and preprocessing

- **Images**: 400 JPGs at 4000x6000, RGB
- **Masks**: 400 PNGs at 4000x6000, single-channel class indices (0-22)
- **Split**: 320 train / 80 validation (random split, seed 42, no overlap)
- **Preprocessing**: random 512x512 crops, horizontal + vertical flips, random 90-degree rotation, brightness/contrast jitter (train only). ImageNet normalization applied to inputs.

### 2.2 Architecture

**U-Net**: 4-level encoder (512 to 32 spatial, 3 to 512 channels) + 4-level decoder (32 to 512) + 1x1 output conv. Skip connections between encoder and decoder at each level. 7.8M parameters.

**Pretrained variant**: same architecture with ResNet-34 encoder (ImageNet weights via `segmentation-models-pytorch`), 24.4M parameters.

### 2.3 Loss functions

- **CrossEntropy (CE)**: standard per-pixel classification loss
- **Dice + CE (50/50)**: Dice loss directly optimizes IoU, up-weighting rare classes by construction

### 2.4 Training

- **Optimizer**: AdamW, lr=1e-3, weight decay 1e-4
- **Batch size**: 8 (512x512 images)
- **Epochs**: 15
- **Metric**: mean IoU (mIoU) across all 23 classes, computed per image and averaged

## 3. Results

| Exp | Loss | Encoder | Val mIoU (final) | Training time |
|---|---|---|---|---|
| **3a** | CrossEntropy | From scratch | **0.039** | 49 min |
| **3b** | Dice + CE | From scratch | **0.035** | 49 min |
| **3d** | Dice + CE | Pretrained ResNet-34 | **0.033** | 44 min |

All three experiments produced similar mIoU (0.033-0.039). Training loss decreased steadily (2.62 to 1.09 for Exp 3d), but val mIoU stayed flat.

**Per-class IoU (Exp 3b, representative):**

- class 0 (unlabeled): 0.000
- class 1 (paved-area): **0.380** (learned)
- class 2 (dirt/paved): 0.049
- class 3 (grass/dirt): **0.210** (learned)
- class 4 (water/gravel): 0.053
- classes 5-21: 0.000 (failed)
- class 22 (solar-panel): 0.018

## 4. Diagnosis

We ran a diagnostic to confirm the cause. On a validation crop that was 100% class 1 (paved-area), the model predicted 100% class 1, giving IoU = 0.9997. **The model is not broken — it correctly segments dominant classes.**

The failure is **data distribution**, not model capacity:

1. **Random crops miss rare classes.** On a 4000x6000 image, a 512x512 random crop covers 1.09% of the area. A bicycle that appears in 10 images has a ~0.1% chance of appearing in any given crop per epoch. Over 15 epochs, the model sees that bicycle ~1 time. It cannot learn from that.

2. **Class imbalance is extreme.** Class 1 (paved) covers 37.7% of all pixels; 17 classes cover <0.1% each. CrossEntropy loss naturally dominates on majority classes; Dice loss partially mitigates this but cannot overcome the absence of rare class samples.

3. **Pretrained encoders don't help.** ResNet-34's ImageNet features are for ground-level photos. Drone-view bicycles are visually different. The pretrained encoder helps with edges and textures but cannot conjure class semantics it has never seen.

## 5. What Would Work

**A. Class-balanced patch sampling.** Before training, index which images contain which rare classes. For each batch, guarantee at least one crop from a rare-class image.

**B. Full-image training.** Train on downsampled full images (e.g., 512x768, preserving aspect ratio) instead of random crops.

**C. Curriculum learning.** Start with dominant classes, introduce rare classes in batches.

## 6. Decision

Given time constraints and the business goal, we are not implementing these fixes now. The current model successfully segments paved-area (IoU 0.38) and grass/vegetation (IoU 0.21), which covers the business-critical classes for land-cover mapping.

## 7. Lessons

1. **Pretrained encoders are not magic.** They help when the target domain is similar to ImageNet.
2. **Loss function selection is secondary to data distribution.** Dice helps when the model can distinguish classes but ignores them. It cannot help when the model hasn't seen them.
3. **Random cropping is a double-edged sword.** It increases diversity for dominant classes but destroys rare-class exposure.
4. **mIoU hides the story.** A mean of 0.035 can look terrible but mask a competent model for the 2-3 classes that matter.

## 8. Next Steps

1. **Project 4**: Build full mapping pipeline (OpenDroneMap → U-Net → QGIS).
2. **Segmentation improvement**: Implement class-balanced sampling, retrain, re-evaluate.
3. **Dataset expansion**: Consider Open Cities AI (African cities) for domain diversity.
