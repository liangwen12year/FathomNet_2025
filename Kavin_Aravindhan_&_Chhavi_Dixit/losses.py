"""
Loss functions used in the FathomNet classification pipeline.

Currently implemented:
----------------------
- FocalLoss: A class-balanced variant of cross-entropy loss.
"""

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for multi-class classification tasks.

    Focal Loss addresses class imbalance by reducing the relative loss
    for well-classified examples and focusing on hard misclassified examples.

    Parameters:
    ----------
    gamma : float
        Focusing parameter that adjusts the rate at which easy examples are down-weighted.
        gamma = 0 reduces to standard cross-entropy.
    alpha : float
        Balancing factor for positive/negative examples.
        Recommended values: 0.25 to 0.5 for imbalanced datasets.
    reduction : str
        Specifies the reduction method to apply to the output:
        - 'mean' : average over batch (default)
        - 'sum'  : sum over batch.
    """
    def __init__(self, gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                target_onehot: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between logits and one-hot encoded targets.

        Parameters:
        ----------
        logits : torch.Tensor of shape [B, C]
            Raw model outputs (before softmax).
        target_onehot : torch.Tensor of shape [B, C]
            One-hot encoded targets. Supports mixed labels (e.g., MixUp).

        Returns:
        ----------
        loss : torch.Tensor (scalar)
            Computed focal loss value.
        """
        # Compute softmax probabilities
        p = torch.softmax(logits, dim=1).clamp_(1e-6, 1 - 1e-6)  # Numerical stability

        # Standard cross-entropy loss (without reduction)
        ce = -(target_onehot * torch.log(p))

        # Focal loss term: down-weight easy examples
        focal_term = self.alpha * (1.0 - p) ** self.gamma * ce

        # Reduction: sum over classes, mean or sum over batch
        if self.reduction == "sum":
            return focal_term.sum()
        else:
            return focal_term.sum(dim=1).mean()
