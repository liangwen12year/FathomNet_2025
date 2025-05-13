# Advance Hierarchical Classification of Ocean Life (FathomNet 2025)

This repository contains all code used in our NeurIPS-style paper *“Advance Hierarchical Classification of Ocean Life.”*  
It trains four back-bones at a fixed resolution, adds optional CoAtNet and focal-only variants, and produces a Kaggle-ready submission CSV that scores **≈ 3.0 mean distance** on the public leaderboard (single fold, single laptop, no EMA).

---

## 1. Repository Structure

```

.
├ dataset.py              # FathomNetDataset + torchvision transforms
├ losses.py               # focal-loss implementation
├ hier\_loss.py            # 0/1/6 distance matrix + expected-distance loss
├ hier\_loss\_copy.py       # same as hier\_loss.py but with updates for CoAtNet
├ model.py                # ConvNeXt-L, Swin-V2-B, ViT-B/MAE, EffV2-M, CoAtNet
├ train\_all.py            # trains all backbones sequentially (no EMA)
├ train.py                # trains ONE backbone: python train.py --arch effv2m
├ train\_copy.py           # train.py version with updates for CoAtNet
├ predict.py              # builds submission.csv from 1-4 checkpoints
├ predict\_copy.py         # predict.py version updated for CoAtNet ensemble
├ requirements.txt
└ fgvc-comp-2025/
└ data/               # put train/ and test/ folders here

````

---

## 2. Prerequisites

* macOS 14 or Linux (tested on Apple M3 Max and NVIDIA A6000)
* Python ≥ 3.10

### 2.1. Create Environment

```bash
conda create -n fathomnet python=3.12 -y
conda activate fathomnet

# Install core libraries
pip install -r requirements.txt
````

---

## 3. Dataset Download

```bash
kaggle competitions download -c fathomnet-2025 -p fgvc-comp-2025
unzip fgvc-comp-2025/fathomnet-2025.zip -d fgvc-comp-2025/data
```

After extraction, you should have:

```
fgvc-comp-2025/data/train/
fgvc-comp-2025/data/test/
```

---

## 4. Training

### 4.1. Train All Four Backbones (ConvNeXt, Swin-V2-B, ViT-B/MAE, EffV2-M)

```bash
python train_all.py               # 4×15 epochs, 384 px, ~9 h on M3 Max
```

Checkpoints appear in `model_checkpoints/`:

```
convnext_best.pt
effv2m_best.pt
swin_best.pt
vit_best.pt
```

### 4.2. Train a Single Model

```bash
python train.py --arch effv2m        # or convnext | swin | vit
```

### 4.3. Train CoAtNet Variant

```bash
python train_copy.py --arch coatnet
```

### Optional Flags:

* `--lr`, `--epochs`, `--batch` for custom hyperparameters.

### 4.4. What the Script Does

* 80/20 stratified split of train CSV.
* RandAugment, MixUp, CutMix augmentations.
* Focal Loss (γ = 2, α = 0.5) + Hierarchy Distance (λ = 0.1).
* Cosine LR scheduler with warm-up.
* AdamW optimizer, no EMA.
* Best epoch saved by validation distance.

---

## 5. Creating a Kaggle Submission

```bash
python predict.py \
  --size 384 \
  --ckpts checkpoints/convnext_best.pt \
          checkpoints/effv2m_best.pt \
          checkpoints/swin_best.pt \
          checkpoints/vit_best.pt \
  --out submission.csv
```

### For CoAtNet (Updated Script):

```bash
python predict_copy.py --size 384 --ckpts checkpoints/coatnet_best.pt --out submission_coatnet.csv
```

* Averages softmax probabilities across models.
* Generates `submission.csv` with `annotations_id,concept_name` format.

---

## 6. Models in This Repository

| Key (`--arch`) | Back-bone (timm name)                              | Params | Notes                 |
| -------------- | -------------------------------------------------- | ------ | --------------------- |
| `effv2m`       | `efficientnetv2_rw_m`                              | 118 M  | fastest convergence   |
| `convnext`     | `convnext_large_in22k`                             | 196 M  | high single-model acc |
| `swin`         | `swinv2_base_window12to24_192to384.ms_in22k`       | 88 M   | complementary errors  |
| `vit`          | `vit_base_patch16_384.mae`                         | 86 M   | MAE pre-training      |
| `coatnet`      | `coatnet_0_rw_224.sw_in1k`                         | 25 M   | used in “coatnet” run |
| any new model  | add subclass to `model.py`, update `ARCH2CLS` dict | –      | plug-and-play         |

---

## 7. Results Snapshot

| Experiment (log)                   | Back-bone(s)                    | Val Acc    | Val Dist | Kaggle   |
| ---------------------------------- | ------------------------------- | ---------- | -------- | -------- |
| `fathomnet_initial`                | ConvNeXt-L                      | 86.06 %    | 1.74     | 3.98     |
| `fathomnet_hier_loss`              | ConvNeXt-L + hierarchy          | 86.73 %    | 1.33     | 3.43     |
| `fathomnet_heir_loss_with_focal`   | ConvNeXt-L + focal              | 86.41 %    | 1.54     | 3.76     |
| `fathomnet_ema_schedular`          | ConvNeXt-L + EMA                | 85.95 %    | 0.38     | 8.55     |
| `fathomnet_heir_loss_with_coatnet` | CoAtNet-0                       | 86.41 %    | 0.18     | 8.71     |
| **4-model ensemble (this repo)**   | EffV2-M + ConvNeXt + Swin + ViT | **88.03 %** | **0.72** | ***** |

---

## 8. Troubleshooting

| Issue                        | Fix                                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: 'xxx'` | Run `pip install -r requirements.txt`                                                              |
| CUDA out of memory           | Reduce `--batch` size or train fewer backbones                                                     |
| Public score ≫ 6             | Ensure predicted labels match official 79-class list (handled in predict.py)                       |
| Need WoRMS-exact metric      | Download `taxonomy_map.json` from organizers, place next to `hier_loss.py` (no code change needed) |

---
