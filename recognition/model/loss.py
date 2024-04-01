import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)

        # 减小简单样本对参数的更新，着重学习复杂样本
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()