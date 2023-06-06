import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction="mean", **kwargs):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        _loss = nn.MSELoss(reduction="none")(outputs, targets).sum(dim=2)
        _p = torch.exp(-_loss)
        loss = (1 - _p) ** self.gamma * _loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
