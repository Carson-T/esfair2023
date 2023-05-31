import torch
import torch.nn as nn

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = torch.tensor(alpha).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target].to(self.device)  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1).to(self.device) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1)).to(self.device)  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1).to(self.device)  # 降维，shape=(bs)
        ce_loss = -logpt.to(self.device)  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt).to(self.device)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss