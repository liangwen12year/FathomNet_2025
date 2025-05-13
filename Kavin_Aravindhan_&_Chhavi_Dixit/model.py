"""
Model definitions for FathomNet hierarchical classification.

This file defines multiple backbone models with fine-level classification heads.
All models are ImageNet-22k pretrained and adapted for 79-class marine taxonomy.

Supported architectures:
- ConvNeXt-Large
- Swin-V2 Base
- ViT-Base (MAE fine-tuned)
- EfficientNetV2-M
"""

import torch
import torch.nn as nn
import timm

# ---------------------------------------------------------------------------
# Base Model Class for all backbones
# ---------------------------------------------------------------------------
class _Base(nn.Module):
    """
    Base wrapper for timm backbones with a fine-level classification head.

    Parameters:
    ----------
    backbone_name : str
        Name of the timm model backbone.
    num_fine : int
        Number of fine-grained classes (default 79).
    """
    def __init__(self, backbone_name: str, num_fine: int = 79):
        super().__init__()
        # Load pretrained backbone (exclude classifier head)
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # Add a linear head for fine-level classification
        self.head = nn.Linear(self.backbone.num_features, num_fine)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through backbone and fine-level classifier.

        Parameters:
        ----------
        x : torch.Tensor
            Input image batch [B, 3, H, W].

        Returns:
        ----------
        output : dict
            Dictionary with key "fine" containing logits [B, num_fine].
        """
        feat = self.backbone(x)
        return {"fine": self.head(feat)}

# ---------------------------------------------------------------------------
# Specific Backbone Classes
# ---------------------------------------------------------------------------

class HierConvNeXt(_Base):
    """
    ConvNeXt-Large backbone pretrained on ImageNet-22k.
    """
    def __init__(self, n=79):
        super().__init__("convnext_large_in22k", n)

class HierSwinB(_Base):
    """
    Swin-V2 Base backbone with progressive window scaling.
    Pretrained on ImageNet-22k.
    """
    def __init__(self, n=79):
        super().__init__("swinv2_base_window12to24_192to384.ms_in22k", n)

class HierViTMAE(_Base):
    """
    ViT-Base backbone pretrained with Masked Autoencoding (MAE).
    Fine-tuned for classification.
    """
    def __init__(self, n=79):
        super().__init__("vit_base_patch16_384.mae", n)

class HierEffV2M(_Base):
    """
    EfficientNetV2-M backbone with reweighted scaling.
    Pretrained on ImageNet-22k.
    """
    def __init__(self, n=79):
        super().__init__("efficientnetv2_rw_m", n)
