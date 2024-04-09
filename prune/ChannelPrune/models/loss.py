import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        
        #ce=-1*log Prediction，ce>0
        #ce越小说明预测越准确，ce越大说明预测与真实的差距越大
        logp = self.ce(input, target)
        
        #ce越小p越接近1，说明预测的越准确，
        #通过这个过程，将简单样本的loss降低，着重训练困难样本
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()