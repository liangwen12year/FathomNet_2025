#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FathomNet training with combinations of CrossEntropy, Focal, and TaxonomicDistance losses,
using EfficientNet‑V2 Large backbone, with configurable α for focal and taxo.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms, models
from torchvision.transforms import RandomErasing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# ─── Losses ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class TaxonomicDistanceLoss(nn.Module):
    def __init__(self, D: torch.Tensor):
        super().__init__()
        self.register_buffer("D", D)  # [C, C]

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)   # [B, C]
        d_i   = self.D[targets]            # [B, C]
        return (probs * d_i).sum(dim=1).mean()

# ─── Dataset ────────────────────────────────────────────────────────────────────

class FathomNetDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_encoder=None, is_test=False):
        self.df        = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test   = is_test
        self.paths     = self.df["path"].tolist()
        if not is_test:
            labels = self.df["label"].tolist()
            if label_encoder is None:
                self.le  = LabelEncoder()
                self.ids = self.le.fit_transform(labels)
            else:
                self.le  = label_encoder
                self.ids = self.le.transform(labels)

    def __len__(self):
        return len(self.paths)

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
                 num_classes: int,
                 lr: float,
                 weight_decay: float,
                 loss_type: str = "ce",
                 gamma: float = 2.0,
                 alpha_focal: float = 1.0,
                 alpha_taxo: float = 1.0,
                 distance_matrix_path: str = None):
        super().__init__()
        self.save_hyperparameters()

        # EfficientNet‑V2 Large backbone
        self.backbone = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_feats, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes)
        )

        # instantiate losses
        self.ce_loss    = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(
            gamma=self.hparams.gamma,
            alpha=self.hparams.alpha_focal
        )
        if self.hparams.distance_matrix_path:
            D = np.load(self.hparams.distance_matrix_path)
            self.taxo_loss = TaxonomicDistanceLoss(torch.tensor(D, dtype=torch.float32))
        else:
            self.taxo_loss = None

    def forward(self, x):
        return self.backbone(x)

    def compute_loss(self, logits, targets):
        lt = self.hparams.loss_type
        loss = 0.0
        # cross‐entropy
        if "ce" in lt:
            ce = self.ce_loss(logits, targets)
            loss += ce
            self.log("train_ce", ce, on_epoch=True, prog_bar=False)
        # focal
        if "focal" in lt:
            fl = self.focal_loss(logits, targets)
            loss += fl
            self.log("train_focal", fl, on_epoch=True, prog_bar=False)
        # taxonomic
        if "taxo" in lt:
            if self.taxo_loss is None:
                raise ValueError("Taxo loss requested but no distance matrix provided")
            tl = self.taxo_loss(logits, targets)
            loss += self.hparams.alpha_taxo * tl
            self.log("train_taxo", tl, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        loss    = self.compute_loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y    = batch
        logits  = self(x)
        loss    = self.compute_loss(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"}
        }

# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.1,0.05),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.95,1.05)),
        RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # labels
    df          = pd.read_csv(args.train_csv)
    le          = LabelEncoder().fit(df["label"].dropna())
    num_classes = len(le.classes_)

    # dataset + split
    full      = FathomNetDataset(args.train_csv, transform=train_tf, label_encoder=le)
    targets   = full.ids
    splitter  = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))

    train_ds  = Subset(full, train_idx)
    val_ds    = Subset(full, val_idx)
    val_ds.dataset.transform = val_tf

    # weighted sampler
    targs      = [targets[i] for i in train_idx]
    counts     = pd.Series(targs).value_counts().sort_index().tolist()
    class_wts  = [1.0 / c for c in counts]
    sample_wts = [class_wts[t] for t in targs]
    sampler    = WeightedRandomSampler(sample_wts, num_samples=len(sample_wts), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = FathomNetClassifier(
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        gamma=args.gamma,
        alpha_focal=args.alpha_focal,
        alpha_taxo=args.alpha_taxo,
        distance_matrix_path=args.distance_matrix
    )

    # callbacks & logger
    ckpt   = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")
    es     = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    lr_mon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("tb_logs", name="fathomnet")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[ckpt, es, lr_mon],
        logger=logger
    )

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume_ckpt or None
    )

    # load best + predict
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_path = ckpt.best_model_path
    model     = FathomNetClassifier.load_from_checkpoint(
        best_path,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        gamma=args.gamma,
        alpha_focal=args.alpha_focal,
        alpha_taxo=args.alpha_taxo,
        distance_matrix_path=args.distance_matrix
    ).to(device)

    test_ds     = FathomNetDataset(args.test_csv, transform=val_tf, label_encoder=le, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    preds, paths = [], []
    with torch.no_grad():
        for imgs, pths in test_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
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
    p = argparse.ArgumentParser("Train FathomNet with Combined Losses")
    p.add_argument("--train_csv",        type=str, required=True)
    p.add_argument("--test_csv",         type=str, required=True)
    p.add_argument("--output_csv",       type=str, default="submission.csv")
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--epochs",           type=int, default=50)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--loss",             choices=["ce","focal","taxo","ce_taxo","focal_taxo"], default="focal_taxo")
    p.add_argument("--gamma",            type=float, default=2.0,
                   help="Focusing parameter for focal loss")
    p.add_argument("--alpha_focal",      type=float, default=1.0,
                   help="α parameter inside the focal loss")
    p.add_argument("--alpha_taxo",       type=float, default=1.0,
                   help="Weight applied to the taxonomic loss component")
    p.add_argument("--distance_matrix",  type=str, default=None,
                   help="Path to .npy distance matrix for taxo loss")
    p.add_argument("--resume_ckpt",      type=str, default=None,
                   help="Path to checkpoint to resume training from")
    args = p.parse_args()
    main(args)

