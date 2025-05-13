# FathomNet Hierarchical Classification Pipeline

> **Note:** The `command_to_reproduce_experiment` script captures both the exact training command and its console output, providing a provenance log that verifies the authenticity and reproducibility of our baseline.

This repository provides a modular workflow for hierarchical taxonomic classification of marine species using underwater imagery. It includes data downloading, taxonomic distance matrix generation, and training scripts for various backbones.

## Prerequisites

* **Python** ≥ 3.8
* **CUDA** (for GPU training)
* Install required packages:

  ```bash
  pip install torch torchvision pytorch-lightning kagglehub ete3 fathomnet tqdm pandas numpy scikit-learn
  ```

## 1. Download Dataset

Run the provided script to fetch and unzip FathomNet 2025 data via KaggleHub:

```bash
python download_fathomnet_data.py
```

This will log in to KaggleHub, download the competition files, and extract them into a local directory (e.g. `~/.cache/kagglehub/competitions/fathomnet-2025/`).

Then, install additional dependencies required for downloading the full dataset:

```bash
pip install -r ~/.cache/kagglehub/competitions/fathomnet-2025/requirements.txt
```

Use the following commands to download training and test images with their annotations:

```bash
python ~/.cache/kagglehub/competitions/fathomnet-2025/download.py \
  ~/.cache/kagglehub/competitions/fathomnet-2025/dataset_train.json ./train/

python ~/.cache/kagglehub/competitions/fathomnet-2025/download.py \
  ~/.cache/kagglehub/competitions/fathomnet-2025/dataset_test.json ./test/
```

## 2. Generate Taxonomic Distance Matrix

Build a 79×79 taxonomic distance matrix from WoRMS lineage data:

```bash
pip install ete3 fathomnet tqdm pandas numpy
python generate_distance_matrix.py \
  --json_path ~/.cache/kagglehub/competitions/fathomnet-2025/dataset_train.json \
  --output_npy distance_matrix.npy \
  --output_csv distance_matrix.csv
```

This outputs `distance_matrix.npy` and `distance_matrix.csv` for use during training.

## 3. Training Models

### EfficientNet‑V2‑M

```bash
python train_validation_test_EfficientNet_V2_M.py \
  --train_csv train/annotations.csv \
  --test_csv test/annotations.csv \
  --output_csv submission_v2m.csv \
  --loss focal_taxo \
  --gamma 2.0 \
  --alpha_focal 0.1 \
  --alpha_taxo 0.5 \
  --distance_matrix distance_matrix.npy \
  --epochs 18 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

### EfficientNet‑V2‑L

```bash
python train_validation_test_EfficientNet_V2_L.py \
  --train_csv train/annotations.csv \
  --test_csv test/annotations.csv \
  --output_csv submission_v2l.csv \
  --loss focal_taxo \
  --gamma 2.0 \
  --alpha_focal 0.1 \
  --alpha_taxo 0.5 \
  --distance_matrix distance_matrix.npy \
  --epochs 12 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

### ConvNeXt‑Large

```bash
python train_validation_test_ConvNeXt_Large.py \
  --train_csv train/annotations.csv \
  --test_csv test/annotations.csv \
  --output_csv submission_convnext.csv \
  --loss focal_taxo \
  --gamma 2.0 \
  --alpha_focal 0.1 \
  --alpha_taxo 0.5 \
  --distance_matrix distance_matrix.npy \
  --epochs 12 \
  --batch_size 32
```

### 3.1 Reproduce ConvNeXt‑Large Focal Baseline

To reproduce the ConvNeXt‑Large focal‐loss baseline (val. loss = 0.333), run:

```bash
python train_validation_test_ConvNeXt_Large.py \
  --train_csv ./train/annotations.csv \
  --test_csv  ./test/annotations.csv \
  --output_csv submiss
```
