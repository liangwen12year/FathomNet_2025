"""
Inference script for generating FathomNet challenge predictions using CoAtNet ensemble.

Usage examples:
---------------
python predict_copy.py --ckpts path/to/ckpt1.pt path/to/ckpt2.pt --out submission.csv
"""

import argparse
import pathlib
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FathomNetDataset, build_transforms
from model_copy import HierCoAtNet
from taxonomy import build_maps

# --------------------------------------------------------
# Argument Parsing
# --------------------------------------------------------
parser = argparse.ArgumentParser(description="FathomNet CoAtNet Ensemble Prediction")
parser.add_argument("--ckpts", nargs="+", required=True, help="List of model checkpoint paths")
parser.add_argument("--batch", type=int, default=64, help="Batch size for inference")
parser.add_argument("--out", type=str, default="submission.csv", help="Output submission file path")
parser.add_argument("--size", type=int, default=384, help="Input image size (default: 384)")
args = parser.parse_args()

# --------------------------------------------------------
# Device Setup
# --------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# --------------------------------------------------------
# Load Label Mappings
# --------------------------------------------------------
ROOT = pathlib.Path("fgvc-comp-2025")
TRAIN_CSV = ROOT / "data/train/annotations.csv"
TEST_CSV = ROOT / "data/test/annotations.csv"
TEST_DIR = ROOT / "data/test"

name2idx, idx2name, _, _ = build_maps(TRAIN_CSV)

# Load train and test annotations
train_df = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

# Extract annotation_id from file path in test data
df_test["annotation_id"] = df_test["path"].apply(
    lambda p: int(pathlib.Path(p).stem.split("_")[-1])
)

# --------------------------------------------------------
# Prepare Test Dataset and DataLoader
# --------------------------------------------------------
test_ds = FathomNetDataset(TEST_CSV, TEST_DIR, name2idx, use_roi=True,
                           split="val", input_size=args.size)

test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

# --------------------------------------------------------
# Load Ensemble Models
# --------------------------------------------------------
models = []
for ckpt_path in args.ckpts:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = HierCoAtNet()
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    models.append(model)

# --------------------------------------------------------
# Inference Loop
# --------------------------------------------------------
all_preds = []

with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
    for imgs in tqdm(test_loader, desc="Running inference"):
        imgs = imgs.to(DEVICE)
        probs = None

        # Ensemble averaging of softmax outputs
        for model in models:
            logits = model(imgs)["fine"]
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p

        # Average predictions across ensemble models
        probs /= len(models)

        # Get predicted class indices
        pred_idx = probs.argmax(1).cpu().tolist()
        all_preds.extend(pred_idx)

# --------------------------------------------------------
# Build Submission DataFrame
# --------------------------------------------------------
submission = pd.DataFrame({
    "annotation_id": df_test["annotation_id"],
    "concept_name": [idx2name[i] for i in all_preds]
})

# Ensure submission is sorted by annotation_id
submission = submission.sort_values("annotation_id")

# --------------------------------------------------------
# Validate Submission Labels
# --------------------------------------------------------
valid_names = set(train_df["label"].unique())
invalid = set(submission["concept_name"]) - valid_names

if invalid:
    print("Invalid concept_name(s) detected in submission:")
    for name in invalid:
        print(f" - {name}")
    raise ValueError("Submission contains invalid concept_name(s). Please verify predictions.")
else:
    print("All concept_name values are valid.")

# --------------------------------------------------------
# Save Submission File
# --------------------------------------------------------
submission.to_csv(args.out, index=False)
print(f"Submission saved to {args.out} with {len(submission)} entries.")

# --------------------------------------------------------
# Example commands for reference:
# --------------------------------------------------------
# python predict_copy.py --ckpts model_checkpoints/hier_loss/ckpt_epoch20.pt --out submission.csv
# python predict_copy.py --ckpts model_checkpoints/ckpt_epoch18.pt model_checkpoints/ckpt_epoch19.pt model_checkpoints/ckpt_epoch20.pt --out submission.csv
