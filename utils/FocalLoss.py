import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss

        return focal_loss


# 定义预测值和目标标签值
preds = torch.tensor([[0.1, 0.9, 0.3],
                      [0.8, 0.2, 0.6],
                      [0.4, 0.5, 0.7]])
targets = torch.tensor([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1]], dtype=torch.float32)

focal_loss = FocalLoss(gamma=2, alpha=None)
loss = focal_loss(preds, targets)

print(loss)
