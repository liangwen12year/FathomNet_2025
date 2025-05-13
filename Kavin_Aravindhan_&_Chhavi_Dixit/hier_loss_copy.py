import json
import numpy as np
import torch
import pathlib
from functools import lru_cache
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalLoss(nn.Module):
    """
    Custom hierarchical loss function combining fine-grained cross-entropy,
    distance-aware focal terms, and coarse label consistency penalty.

    Parameters:
    ----------
    coarse_of_idx : dict
        Mapping from fine class index to its corresponding coarse class index.
    gamma : float
        Focusing parameter for the distance-based focal term.
    num_classes : int
        Total number of fine-grained classes (default is 79).
    """
    def __init__(self, coarse_of_idx, gamma=2, num_classes=79):
        super().__init__()
        self.coarse_of_idx = coarse_of_idx
        self.gamma = gamma
        
        # Learnable weights for hierarchical and consistency components
        self.register_parameter('h_weight', nn.Parameter(torch.tensor(0.3)))
        self.register_parameter('c_weight', nn.Parameter(torch.tensor(0.2)))
        
        # Precompute fine-to-coarse label mapping and register as a buffer (device-safe)
        self.register_buffer('coarse_mapping', 
            torch.tensor([coarse_of_idx[i] for i in range(num_classes)], dtype=torch.long)
        )

    def forward(self, logits, targets, D):
        """
        Compute the total hierarchical loss.

        Parameters:
        ----------
        logits : Tensor (batch_size x num_classes)
            Raw model outputs (logits) for fine classes.
        targets : Tensor (batch_size,)
            Ground truth fine class indices.
        D : ndarray
            Distance matrix between fine classes.

        Returns:
        ----------
        total_loss : Tensor
            Combined loss value.
        """
        # Fine-grained cross-entropy with label smoothing
        loss_fine = F.cross_entropy(logits, targets, label_smoothing=0.2)
        
        # Retrieve corresponding coarse labels for targets
        coarse_targets = self.coarse_mapping[targets]
        
        # Hierarchical distance-aware focal loss component
        loss_hier = self.focal_distance(logits, targets, D)
        
        # Consistency penalty: penalize fine predictions inconsistent with coarse labels
        loss_consistency = (self.coarse_mapping[logits.argmax(1)] != coarse_targets).float().mean()
        
        # Total loss as weighted sum
        total_loss = loss_fine + self.h_weight * loss_hier + self.c_weight * loss_consistency
        return total_loss

    def focal_distance(self, logits, targets, D):
        """
        Distance-weighted focal loss component based on softmax outputs.

        Parameters:
        ----------
        logits : Tensor (batch_size x num_classes)
            Raw model outputs (logits).
        targets : Tensor (batch_size,)
            Ground truth fine class indices.
        D : ndarray
            Distance matrix between fine classes.

        Returns:
        ----------
        focal_term : Tensor
            Mean focal-distance loss across the batch.
        """
        # Compute softmax probabilities
        P = torch.softmax(logits, dim=1)
        
        # Extract relevant rows from distance matrix for the batch targets
        Dt = torch.from_numpy(D[targets.cpu().numpy()]).to(logits.device).float()
        
        # Normalize and stabilize distance values between 0 and 1
        Dt = (Dt - Dt.min()) / (Dt.max() - Dt.min() + 1e-8)
        Dt = torch.clamp(Dt, min=1e-4, max=1.0)
        
        # Compute focal-distance weighted probabilities
        focal_term = (P * Dt * (1 - Dt)**self.gamma).sum(1).mean()
        return focal_term
