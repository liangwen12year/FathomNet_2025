#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training pipeline for FathomNet combining Focal Loss and TaxonomicDistanceLoss.
Supports loss types: ce, focal, taxo, focal_taxo.
"""
import argparse
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

from torchvision import transforms, models
from torchvision.transforms import RandomErasing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

# ─── Loss Modules ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = "mean"):
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
        probs = F.softmax(logits, dim=1)           # [B, C]
        d_i   = self.D[targets]                    # [B, C]
        return (probs * d_i).sum(dim=1).mean()

class FocalTaxoLoss(nn.Module):
    def __init__(self,
                 D: torch.Tensor,
                 gamma: float = 2.0,
                 alpha_focal: float = None,
                 alpha_taxo: float = 1.0,
                 reduction: str = "mean"):
        super().__init__()
        self.focal     = FocalLoss(gamma=gamma, alpha=alpha_focal, reduction=reduction)
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
        self.data       = pd.read_csv(csv_path)
        self.transform  = transform
        self.is_test    = is_test
        self.paths      = self.data["path"].tolist()

        if not is_test:
            self.labels = self.data["label"].tolist()
            if label_encoder is None:
                self.le  = LabelEncoder()
                self.ids = self.le.fit_transform(self.labels)
            else:
                self.le  = label_encoder
                self.ids = self.le.transform(self.labels)

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
                 train_csv: str,
                 num_classes: int,
                 lr: float,
                 weight_decay: float,
                 loss_type: str,
                 gamma: float,
                 alpha_focal: float,
                 alpha_taxo: float,
                 distance_matrix: str):
        super().__init__()
        self.save_hyperparameters()

        # backbone
        self.backbone = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        )
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_feats, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, self.hparams.num_classes)
        )

        # prepare distance matrix if needed
        D_tensor = None
        if loss_type in ("taxo", "focal_taxo"):
            # load CSV ordering
            csv_path = distance_matrix.replace(".npy", ".csv")
            df_mat   = pd.read_csv(csv_path, index_col=0)
            orig_names = list(df_mat.index)
            orig_mat   = df_mat.values.astype(np.float32)

            # reorder to match train Csv LabelEncoder
            df_train = pd.read_csv(train_csv)
            le       = LabelEncoder().fit(df_train["label"].dropna())
            target_names = list(le.classes_)
            perm     = [orig_names.index(n) for n in target_names]
            D_reord  = orig_mat[np.ix_(perm,perm)]
            D_reord /= D_reord.max()
            D_tensor = torch.tensor(D_reord, dtype=torch.float32)

        # select loss
        lt = loss_type
        if lt == "ce":
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        elif lt == "focal":
            self.criterion = FocalLoss(gamma=gamma, alpha=alpha_focal)
        elif lt == "taxo":
            self.criterion = TaxonomicDistanceLoss(D_tensor)
        elif lt == "focal_taxo":
            self.criterion = FocalTaxoLoss(
                D_tensor,
                gamma=gamma,
                alpha_focal=alpha_focal,
                alpha_taxo=alpha_taxo
            )
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
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        sched       = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

# ─── Main Training Script ──────────────────────────────────────────────────────

def main(args):
    # transforms
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

    df       = pd.read_csv(args.train_csv)
    le       = LabelEncoder().fit(df["label"].dropna())
    num_cls  = len(le.classes_)

    full     = FathomNetDataset(args.train_csv, transform=train_tf, label_encoder=le)
    targets  = full.ids
    splitter = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(full, val_idx)
    val_ds.dataset.transform = val_tf

    # weighted sampler
    targs       = [targets[i] for i in train_idx]
    counts      = pd.Series(targs).value_counts().sort_index().tolist()
    class_wts   = [1.0/c for c in counts]
    samp_wts    = [class_wts[t] for t in targs]
    sampler     = WeightedRandomSampler(samp_wts, num_samples=len(samp_wts), replacement=True)

    train_loader= DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader  = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = FathomNetClassifier(
        train_csv=args.train_csv,
        num_classes=num_cls,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        gamma=args.gamma,
        alpha_focal=args.alpha_focal,
        alpha_taxo=args.alpha_taxo,
        distance_matrix=args.distance_matrix
    )

    ckpt   = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")
    es     = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lr_mon = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger("tb_logs", name="fathomnet")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16,
        gradient_clip_val=1.0,
        callbacks=[ckpt, es, lr_mon],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    # load best and predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best   = FathomNetClassifier.load_from_checkpoint(
        ckpt.best_model_path,
        train_csv=args.train_csv,
        num_classes=num_cls,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        gamma=args.gamma,
        alpha_focal=args.alpha_focal,
        alpha_taxo=args.alpha_taxo,
        distance_matrix=args.distance_matrix
    ).to(device)

    test_ds     = FathomNetDataset(args.test_csv, transform=val_tf, label_encoder=le, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    best.eval()
    preds, paths = [], []
    with torch.no_grad():
        for imgs, pths in test_loader:
            imgs = imgs.to(device)
            out  = best(imgs)
            idxs = torch.argmax(out, dim=1).cpu().tolist()
            preds.extend(idxs)
            paths.extend(pths)

    decoded    = le.inverse_transform(preds)
    submission = pd.read_csv(args.test_csv)
    submission["annotation_id"] = range(1, len(submission)+1)
    submission["concept_name"]  = decoded
    submission.drop(["path","label"], axis=1).to_csv(args.output_csv, index=False)

    print(f"Saved submission: {args.output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("FathomNet Focal+Taxo Training")
    p.add_argument("--train_csv",        type=str, required=True)
    p.add_argument("--test_csv",         type=str, required=True)
    p.add_argument("--output_csv",       type=str, default="submission.csv")
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--epochs",           type=int, default=50)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--loss",             choices=["ce","focal","taxo","focal_taxo"], default="focal_taxo")
    p.add_argument("--gamma",            type=float, default=2.0, help="Focal gamma")
    p.add_argument("--alpha_focal",      type=float, default=None, help="Focal alpha")
    p.add_argument("--alpha_taxo",       type=float, default=1.0, help="Taxo weight")
    p.add_argument("--distance_matrix",  type=str, required=False, help="Path to distance_matrix.npy/.csv")
    args = p.parse_args()
    main(args)

