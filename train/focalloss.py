from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""FocalLoss"""
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', **kwargs):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, outputs, targets):
        # import pdb; pdb.set_trace()
        # loss = sigmoid_focal_loss(outputs, targets, self.alpha, self.gamma, self.reduction)

        # p = torch.sigmoid(outputs)
        # ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
        # p_t = p * targets + (1 - p) * (1 - targets)
        # loss = ce_loss * ((1 - p_t) ** self.gamma)

        # if self.alpha >= 0:
            # alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            # loss = alpha_t * loss

        _loss = nn.MSELoss(reduction='none')(outputs, targets).sum(dim=2)
        p = torch.exp(-_loss)
        loss = (1 - p) ** self.gamma * _loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss