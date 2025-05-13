"""
Model architectures for hierarchical classification in FathomNet.

Each model has:
- A shared feature backbone (pretrained on ImageNet)
- Two classification heads:
    - Fine-grained head (species level)
    - Coarse-grained head (family/genus level)

Supported architectures:
- ConvNeXt-Large
- CoAtNet (Hybrid CNN-Transformer)
- MaxViT (Scalable Vision Transformer)
- Swin-V2 Base
- ViT-Base (MAE fine-tuned)
"""

import torch
import torch.nn as nn
import timm

# ---------------------------------------------------------------------------
# Base Model: ConvNeXt + Fine and Coarse Heads
# ---------------------------------------------------------------------------
class HierConvNeXt(nn.Module):
    """
    Hierarchical classifier using ConvNeXt or any compatible backbone.

    Parameters:
    ----------
    num_fine : int
        Number of fine-grained classes (default: 79).
    num_coarse : int
        Number of coarse-grained classes (default: 12).
    backbone : str
        timm model backbone name.
    """
    def __init__(self,
                 num_fine: int = 79,
                 num_coarse: int = 12,
                 backbone: str = "convnext_large_in22k"):
        super().__init__()
        # Load pretrained backbone (without classifier)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        in_feats = self.backbone.num_features

        # Fine-level classifier head
        self.head_fine = nn.Linear(in_feats, num_fine)

        # Coarse-level classifier head
        self.head_coarse = nn.Linear(in_feats, num_coarse)

    def forward(self, x):
        """
        Forward pass through backbone and both classifier heads.

        Returns:
        ----------
        Dict with 'fine' and 'coarse' logits.
        """
        feat = self.backbone(x)
        return {
            "fine": self.head_fine(feat),
            "coarse": self.head_coarse(feat)
        }

# ---------------------------------------------------------------------------
# CoAtNet Variant (Hybrid CNN-Transformer)
# ---------------------------------------------------------------------------
class HierCoAtNet(HierConvNeXt):
    """
    CoAtNet variant with added dropout regularization and a bottleneck in fine head.

    Parameters:
    ----------
    num_fine : int
        Number of fine-grained classes.
    num_coarse : int
        Number of coarse-grained classes.
    img_size : int
        Input image resolution (affects backbone selection).
    """
    def __init__(self, num_fine=79, num_coarse=12, img_size=224):
        super().__init__(
            num_fine=num_fine,
            num_coarse=num_coarse,
            backbone=f"coatnet_rmlp_1_rw_{img_size}.sw_in1k"
        )

        # Dropout regularization for better generalization
        self.dropout = nn.Dropout2d(0.5)

        # Fine head with bottleneck and dropout
        self.head_fine = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            self.dropout,
            nn.Linear(512, num_fine)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "fine": self.head_fine(feat),
            "coarse": self.head_coarse(feat)
        }

# ---------------------------------------------------------------------------
# MaxViT Variant
# ---------------------------------------------------------------------------
class HierMaxViT(HierConvNeXt):
    """
    MaxViT backbone with hierarchical token attention.
    """
    def __init__(self, num_fine=79, num_coarse=12):
        super().__init__(num_fine, num_coarse,
            backbone="maxvit_base_tf_224.in21k")

# ---------------------------------------------------------------------------
# Swin-V2 Base Variant
# ---------------------------------------------------------------------------
class HierSwinB(HierConvNeXt):
    """
    Swin-V2 Base backbone pretrained on ImageNet-22k.
    Uses shifted window self-attention with progressive scaling.
    """
    def __init__(self, num_fine=79, num_coarse=12):
        super().__init__(num_fine, num_coarse,
            backbone="swinv2_base_window12to24_192to384.ms_in22k")

# ---------------------------------------------------------------------------
# ViT-Base MAE-finetuned Variant
# ---------------------------------------------------------------------------
class HierViTMAE(HierConvNeXt):
    """
    ViT-Base backbone pretrained with Masked Autoencoding (MAE).
    Fine-tuned for image classification tasks.
    """
    def __init__(self, num_fine=79, num_coarse=12):
        super().__init__(num_fine, num_coarse,
            backbone="vit_base_patch16_384.mae")
