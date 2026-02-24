"""
Loss functions for wall segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal loss for dense segmentation.

    Multiplies per-pixel cross-entropy by (1 - p)^gamma so that easy, already-correct
    predictions contribute almost no gradient and hard, uncertain predictions dominate.

    FL(p) = -(1 - p_correct)^gamma * log(p_correct)

    Args:
        weight:  Per-class weights (same as nn.CrossEntropyLoss weight). Applied before
                 the focal factor so the two mechanisms are complementary.
        gamma:   Focusing parameter. 0 = standard CE, 2 = standard focal (default).
        reduction: 'mean' (default) | 'sum' | 'none'

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        # Keep weight on CPU to avoid CUDA IPC issues with DataLoader workers on Windows.
        # It is moved to the correct device lazily in forward().
        self._weight_cpu = weight.cpu() if weight is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  (B, C, H, W) raw logits
            targets: (B, H, W) integer class labels
        Returns:
            Scalar loss (or per-pixel tensor if reduction='none')
        """
        weight = self._weight_cpu.to(inputs.device) if self._weight_cpu is not None else None
        # Per-pixel CE loss (unreduced) — shape (B, H, W)
        ce = F.cross_entropy(inputs, targets, weight=weight, reduction='none')

        # p_correct = e^{-CE}  (probability assigned to the correct class)
        pt = torch.exp(-ce)

        # Focal factor
        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal
