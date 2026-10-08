# EuroSAT Land-Use Classification with ResNet-18

Project 1 in my series for the NPTEL / IISc course **AI-driven Perception, Learning and Mapping for Drones**
(Module 1: foundations of neural networks, Module 2: CNNs).

I classify satellite patches from the EuroSAT dataset into 10 land-use classes, then run controlled experiments
(one change at a time) to understand what actually moves accuracy: training length, learning-rate schedule,
layer freezing, data augmentation, and pretraining.

**Best result: 97.8% test accuracy** (pretrained ResNet-18 + learning-rate schedule).

## Results

Each row changes one thing relative to the row above or to Experiment 2 (see the notebook for details).
"Original" is the first logged run; "Rerun" is a fresh top-to-bottom run of the cleaned notebook on a Colab T4.

| # | Experiment | Original | Rerun |
|---|---|---|---|
| 0 | Baseline: pretrained ResNet-18, 10 epochs | 93.6% | 94.1% |
| 1 | 20 epochs | 94.6% | 95.4% |
| 2 | + StepLR learning-rate schedule | **97.8%** | 97.5% |
| 3 | Freeze conv1..layer2 (~6% of params) | 96.7% | 96.8% |
| 3b | Freeze conv1..layer3 (~25% of params) + BatchNorm-eval fix | 95.7% | 96.0% |
| 4 | Remove augmentation | 96.5% | 97.0% |
| 5 | Small CNN from scratch (586K params) | 96.2% | 96.1% |

## Key findings

1. **The learning-rate schedule was the biggest lever.** Adding StepLR gave roughly +2 to +3% on its own,
   clearly beyond run-to-run noise.
2. **Augmentation prevents memorization.** Without it, training accuracy reached 100% and the train/validation
   gap grew about 9x. The test-accuracy cost was small and within noise on a rerun, so the gap, not the score,
   is the clearer evidence.
3. **Pretraining mostly buys speed on this easy dataset.** A 586K-parameter CNN trained from scratch came within
   about 1.6% of the pretrained ResNet-18 (11.2M parameters), but converged much more slowly.
4. **Freezing is a trade-off.** Accuracy dropped as more of the network was frozen.

## Caveats

- Each configuration was run once per pass, and the original runs seeded only the data split.
- With a 4,050-image test set, differences of about 1% are suggestive rather than proven.
- Experiments 3 and 3b differ in two ways (an extra frozen layer and the BatchNorm-eval fix), so that comparison
  is not a clean single-variable test.
- Experiments 0-1 were originally run on CPU, 2-5 on GPU. The `logs/` folder holds the original records.

## Setup

- **Data:** EuroSAT RGB, 27,000 Sentinel-2 patches (64x64, 10 classes), split 70/15/15 with seed 42
  (18,900 / 4,050 / 4,050). Downloaded automatically by `torchvision.datasets.EuroSAT`.
- **Model:** ImageNet-pretrained ResNet-18 with a new 10-class head (Exp 5 uses a small custom CNN).
- **Training:** Adam (lr 1e-3), cross-entropy, batch size 64, 20 epochs (10 for Exp 0).
- **Augmentation:** random horizontal flip, vertical flip, and rotation (up to 90 degrees).

## Repository contents

```
EuroSAT_ResNet18.ipynb   # the full notebook, run top to bottom
logs/                    # original per-experiment logs (.txt)
README.md
```

Model checkpoints (`.pt`) are not included; the notebook recreates them.

## Running it

1. Open `EuroSAT_ResNet18.ipynb` in Google Colab.
2. Set the runtime to GPU (Runtime, Change runtime type, T4).
3. Run all (about 25 minutes). The notebook mounts Google Drive and writes checkpoints and logs to a `eurosat` folder.
   Set `TAG = "_rerun"` in the setup cell if you want to keep earlier outputs.

Outside Colab, the notebook falls back to a local `./eurosat_outputs` folder.

## What's next

- Object detection on aerial imagery (VisDrone)
- Segmentation on RGB drone imagery, moving toward map-style outputs
- Optional: multi-spectral EuroSAT (13 bands) with `torchgeo`

## Citation

Dataset: Helber, Bischke, Dengel, Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

## Author

Mark David Mapalo Mwila
