import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothLoss(nn.Module):
    def __init__(self, weight, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, targets):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        smooth_loss = -log_preds.sum(dim=-1).mean() if self.reduction == "mean" else -log_preds.sum(dim=-1).sum()
        nll_loss = F.nll_loss(log_preds, targets, self.weight, reduction=self.reduction)
        return (1 - self.epsilon) * nll_loss + self.epsilon * (smooth_loss / n)
