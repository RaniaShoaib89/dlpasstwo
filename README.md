# Deep Learning for Perception: Assignment 2: Section B

**Roll Number:** 23F-0650  
**Assignment:** DL_ASS02  
**Repository:** https://github.com/RaniaShoaib89/dlpasstwo

---

## Overview

This repository contains my code and results for Section B of Assignment 2 (CIFAR-10 classification with limited training data).

The assignment tests how well a small convolutional network handles severe distribution shifts when trained on very few samples:
- **Small Training Set:** 500 clean CIFAR-10 images (50 per class), split into 400 training and 100 validation images using random seed `42`.
- **Corrupted Test Set:** 10,000 CIFAR-10 test images passed through a seeded corruption pipeline:
  - Random rotation up to ±25°
  - Brightness and contrast shifts (between 0.6× and 1.4×)
  - Random rectangular occlusion covering ~25% of the image (16×16 patch)
  - Defocus Gaussian blur as the hidden corruption
- **Rules:**
  - Train from scratch without pretrained weights.
  - Parameter count under 1,000,000 (my model uses 365,290).
  - Must include an Inception-style module with parallel 1×1, 3×3, 5×5, and pooling branches.
  - Up to 60 epochs per run, evaluated across three random seeds (`42`, `123`, `999`).
  - No training or validation on either test set.

---

## Files

```
├── DL_ASS02_23F_0650.ipynb        # Standalone Jupyter notebook (executed with visible outputs and plots)
├── DL_ASS02_23F_0650.docx         # Word report with results table, answers, and plots
└── README.md                      # Instructions to run the code
```

---

## How to Run

The notebook [`DL_ASS02_23F_0650.ipynb`](./DL_ASS02_23F_0650.ipynb) is self-contained and needs no extra local Python files.

### In Google Colab (TPU / GPU)
1. Open [Google Colab](https://colab.research.google.com/) and upload `DL_ASS02_23F_0650.ipynb`.
2. Go to **Runtime > Change runtime type** and pick **TPU** (or **GPU**).
3. Click **Runtime > Run all**. The notebook auto-detects Colab TPU (`torch_xla`), GPU (`cuda`), or CPU. All 21 runs finish in under two minutes on TPU or GPU.

### Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib

jupyter notebook DL_ASS02_23F_0650.ipynb
```

---

## Ablation Study Results (Seeds 42, 123, 999)

Each configuration was trained on 400 images across three seeds, reporting mean and standard deviation:

| Configuration | Validation Acc (%) | Clean Test Acc (%) | Shifted Test Acc (%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **1. No augmentation (Baseline)** | 52.0 ± 1.4 | 43.2 ± 1.1 | 13.6 ± 0.4 | Hits 100% train accuracy; fails on shifted inputs |
| **2. Geometric only** | 54.3 ± 0.5 | 42.4 ± 1.5 | 13.6 ± 0.3 | Flips, crops, and ±25° rotation improve validation score |
| **3. Photometric only** | 54.3 ± 2.4 | 40.8 ± 0.9 | 15.7 ± 1.3 | Color/contrast jitter helps against lighting shifts (+2.1%) |
| **4. Geometric + Photometric** | 51.0 ± 2.9 | 40.4 ± 2.8 | 16.0 ± 0.4 | Combines spatial and lighting transforms |
| **5. + Cutout / Random Erasing** | 47.3 ± 2.1 | 36.5 ± 0.3 | **17.0 ± 2.7** | **Best gain (+3.43% over baseline)**; handles 25% occlusion |
| **6. + Mixup + Label Smoothing** | 43.3 ± 1.7 | 33.5 ± 2.0 | 15.3 ± 0.9 | Mixup and 0.1 label smoothing smooth decision boundaries |
| **7. Final Model** | 43.3 ± 1.7 | 33.5 ± 2.0 | 15.3 ± 0.9 | Full combined augmentation pipeline |

---

## Main Takeaways

1. **Baseline Failure (Part A):**  
   The baseline reaches 100.00% train accuracy but drops to 44.09% on clean test and 13.89% on shifted test. With only 400 clean images, the model memorizes exact pixel values. When evaluated on rotated, occluded, and blurred images, predictions drop to near random chance.

2. **Single Most Helpful Change (Part D):**  
   Cutout gave the biggest jump on shifted test data (+3.43% over baseline, reaching 16.99 ± 2.74%). Since the shifted test set covers ~25% of each image with a blank rectangle, standard CNNs fail whenever their main feature gets covered. Cutout forces the network to use multiple parts of the image instead.

3. **Augmentation Trade-offs (Part B):**  
   - *Vertical Flip:* Hurts CIFAR-10 because cars, planes, and animals naturally stay upright under gravity. Flipping them upside down teaches useless orientations.  
   - *±90° vs ±25° Rotation:* Rotating ±90° wastes model capacity learning extreme angles absent from the test set, hurting performance on normal upright images.  
   - *Too Much Augmentation:* Stacking too many heavy distortions on 400 images destroys recognizable features, causing small models to underfit.

4. **Regularization (Part C):**  
   Dropout (0.4) is placed after Global Average Pooling and before the final linear layer. Dropping pixels between early convolutional layers does little because adjacent pixels share spatial info. After GAP, each feature is a single summary number for the whole image, where dropout effectively prevents co-adaptation.

5. **Error Analysis (Part E):**  
   - *Most Confused:* Cat vs. Dog (similar fur, ears, and quadruped shape at 32×32) and Automobile vs. Truck (when the truck bed gets occluded).  
   - *Hidden Corruption:* The model reached 35.03% on unlisted defocus blur. Training with mild blur helped the network learn basic silhouettes instead of relying strictly on sharp pixel edges.

6. **Test-Time Augmentation (Part F):**  
   Standard shifted accuracy was 14.78% (12.88s) compared to 14.33% with 4-way TTA (46.81s, a 3.63× slowdown). Under heavy rectangular occlusion, flips and rolls cannot bring back missing pixels, so standard inference is faster and slightly more accurate.
