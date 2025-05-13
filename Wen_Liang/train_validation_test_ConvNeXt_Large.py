#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FathomNet training with ConvNeXt‑Large backbone and flexible losses:
  - ce
  - focal
  - taxo
  - ce_taxo
  - focal_taxo

Switches to precision="16-mixed" AMP.
"""

import argparse, os, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.transforms import RandomErasing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# ─── Loss Modules ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = (self.alpha or 1.0) * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class TaxonomicDistanceLoss(nn.Module):
    def __init__(self, D: torch.Tensor):
        super().__init__()
        self.register_buffer("D", D)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)    # [B,C]
        d_i   = self.D[targets]             # [B,C]
        return (probs * d_i).sum(dim=1).mean()

class CombinedTaxoCE(nn.Module):
    """ CE + α_taxo × Taxo """
    def __init__(self, D: torch.Tensor, smoothing=0.1, alpha_taxo=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.register_buffer("D", D)
        self.alpha_taxo = alpha_taxo

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probs   = F.softmax(logits, dim=1)
        d_i     = self.D[targets]
        taxo    = (probs * d_i).sum(dim=1).mean()
        return ce_loss + self.alpha_taxo * taxo

class FocalTaxoLoss(nn.Module):
    """ Focal + α_taxo × Taxo """
    def __init__(self, D: torch.Tensor, gamma=2.0, alpha_focal=None, alpha_taxo=1.0, reduction="mean"):
        super().__init__()
        self.focal      = FocalLoss(gamma=gamma, alpha=alpha_focal, reduction=reduction)
        self.register_buffer("D", D)
        self.alpha_taxo = alpha_taxo

    def forward(self, logits, targets):
        fl_loss = self.focal(logits, targets)
        probs   = F.softmax(logits, dim=1)
        d_i     = self.D[targets]
        taxo    = (probs * d_i).sum(dim=1).mean()
        return fl_loss + self.alpha_taxo * taxo

# ─── Dataset ────────────────────────────────────────────────────────────────────

class FathomNetDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_encoder=None, is_test=False):
        self.data      = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test   = is_test
        self.paths     = self.data["path"].tolist()
        if not is_test:
            labels = self.data["label"].tolist()
            if label_encoder is None:
                self.le  = LabelEncoder()
                self.ids = self.le.fit_transform(labels)
            else:
                self.le  = label_encoder
                self.ids = self.le.transform(labels)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, self.paths[idx]
        return img, self.ids[idx]

# ─── Lightning Module ──────────────────────────────────────────────────────────

class FathomNetClassifier(pl.LightningModule):
    def __init__(self,
                 train_csv: str,
                 num_classes: int,
                 lr: float,
                 weight_decay: float,
                 loss_type: str,
                 gamma: float,
                 alpha_focal: float,
                 alpha_taxo: float,
                 distance_matrix: str = None):
        super().__init__()
        self.save_hyperparameters()

        # ConvNeXt-Large backbone
        self.backbone = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        in_feats = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_feats, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, self.hparams.num_classes),
        )

        # Preload taxo matrix if needed
        D = None
        if loss_type in ("taxo", "ce_taxo", "focal_taxo"):
            mat = np.load(self.hparams.distance_matrix)
            D   = torch.tensor(mat, dtype=torch.float32)

        lt = loss_type
        if lt == "ce":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif lt == "focal":
            self.criterion = FocalLoss(gamma=self.hparams.gamma,
                                       alpha=self.hparams.alpha_focal)
        elif lt == "taxo":
            self.criterion = TaxonomicDistanceLoss(D)
        elif lt == "ce_taxo":
            self.criterion = CombinedTaxoCE(D,
                                            smoothing=0.1,
                                            alpha_taxo=self.hparams.alpha_taxo)
        elif lt == "focal_taxo":
            self.criterion = FocalTaxoLoss(D,
                                           gamma=self.hparams.gamma,
                                           alpha_focal=self.hparams.alpha_focal,
                                           alpha_taxo=self.hparams.alpha_taxo)
        else:
            raise ValueError(f"Unknown loss_type {lt}")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        loss    = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        loss    = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.1,0.05),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.95,1.05)),
        RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Load & split
    df      = pd.read_csv(args.train_csv)
    le      = LabelEncoder().fit(df["label"].dropna())
    num_cls = len(le.classes_)

    full     = FathomNetDataset(args.train_csv, transform=train_tf, label_encoder=le)
    targets  = full.ids
    splitter = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(full, val_idx)
    val_ds.dataset.transform = val_tf

    # Weighted sampler
    targs      = [targets[i] for i in train_idx]
    counts     = pd.Series(targs).value_counts().sort_index().tolist()
    class_wts  = [1.0/c for c in counts]
    sample_wts = [class_wts[t] for t in targs]
    sampler    = WeightedRandomSampler(sample_wts, len(sample_wts), True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = FathomNetClassifier(
        train_csv       = args.train_csv,
        num_classes     = num_cls,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        loss_type       = args.loss,
        gamma           = args.gamma,
        alpha_focal     = args.alpha_focal,
        alpha_taxo      = args.alpha_taxo,
        distance_matrix = args.distance_matrix
    )

    ckpt   = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")
    es     = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_mon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("tb_logs", name="fathomnet")

    trainer = pl.Trainer(
        max_epochs       = args.epochs,
        accelerator      = "gpu",
        devices          = 1,               # single‐GPU
        precision        = "16-mixed",      # modern AMP
        gradient_clip_val= 1.0,
        callbacks        = [ckpt, es, lr_mon],
        logger           = logger
    )

    trainer.fit(model, train_loader, val_loader)

    # ── Best‐model inference ────────────────────────────────────────
    device     = torch.device("cuda")
    best_model = FathomNetClassifier.load_from_checkpoint(
        ckpt.best_model_path,
        train_csv       = args.train_csv,
        num_classes     = num_cls,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        loss_type       = args.loss,
        gamma           = args.gamma,
        alpha_focal     = args.alpha_focal,
        alpha_taxo      = args.alpha_taxo,
        distance_matrix = args.distance_matrix
    ).to(device)

    test_ds     = FathomNetDataset(args.test_csv, transform=val_tf, label_encoder=le, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    best_model.eval()
    preds, paths = [], []
    with torch.no_grad():
        for imgs, pths in test_loader:
            imgs = imgs.to(device)
            out  = best_model(imgs)
            idxs = torch.argmax(out, dim=1).cpu().tolist()
            preds.extend(idxs)
            paths.extend(pths)

    decoded = le.inverse_transform(preds)
    sub     = pd.read_csv(args.test_csv)
    sub["annotation_id"] = range(1, len(sub)+1)
    sub["concept_name"]  = decoded
    sub.drop(["path","label"], axis=1).to_csv(args.output_csv, index=False)

    print(f"Saved submission: {args.output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("FathomNet ConvNeXt‑Large Combined Loss")
    p.add_argument("--train_csv",       type=str, required=True)
    p.add_argument("--test_csv",        type=str, required=True)
    p.add_argument("--output_csv",      type=str, default="submission.csv")
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--epochs",          type=int, default=50)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--loss",
                   choices=["ce","focal","taxo","ce_taxo","focal_taxo"],
                   default="ce",
                   help="Which loss to use")
    p.add_argument("--gamma",           type=float, default=2.0,
                   help="Focusing parameter for focal loss")
    p.add_argument("--alpha_focal",     type=float, default=None,
                   help="Alpha for focal loss")
    p.add_argument("--alpha_taxo",      type=float, default=1.0,
                   help="Weight on the taxo term")
    p.add_argument("--distance_matrix", type=str, default=None,
                   help="Path to .npy distance matrix")
    args = p.parse_args()
    main(args)

