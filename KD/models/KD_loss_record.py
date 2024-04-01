import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import FocalLoss
from models.metric import ArcFace, CosFace
from config import config as conf

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class KD_loss(nn.Module):
    def __init__(self, class_num, T=1, alpha=0.5, beta=1.0, gamma=1.0, embedding_size=512):
        super().__init__()
        self.T = T
        # hard_loss 
        self.alpha = alpha
        # soft_loss_based_on_feature
        self.beta = beta
        # classification loss 
        self.delta = 0.9
        # soft label loss
        self.sigma = 0.1
        self.criterion = FocalLoss().to(conf.device)
        self.arcface = ArcFace(embedding_size=embedding_size, class_num=class_num).to(conf.device)
    
    def forward (self, teacher_outputs, student_outputs, labels):

        # # feature distillation loss
        # T_t=(teacher_outputs/self.T).softmax(dim=1)
        # T_s=(student_outputs/self.T).softmax(dim=1)
        # inter_loss = self.T**2 * inter_class_relation(T_s, T_t)
        # intra_loss = self.T**2 * intra_class_relation(T_s, T_t)
        # # dist_loss = self.beta * inter_loss + self.gamma * intra_loss

        # classification loss - hard label \ hard loss
        thetas = self.arcface(student_outputs, labels)
        cls_loss = self.criterion(thetas, labels)

        thetas_tea = self.arcface(teacher_outputs, labels)
        thetas_stu = self.arcface(student_outputs, labels)

        # soft label loss
        #将此处改为cosine_similarity
        #经过调试发现，teacher_predict_label的值预测准确率极低，而且num_classes=10575，所以teacher_predict_label的值都非常接近0
        # student_predict_label = F.linear(F.normalize(student_outputs), F.normalize(self.arcface.weight))
        # teacher_predict_label = F.linear(F.normalize(teacher_outputs), F.normalize(self.arcface.weight))
        # 计算两个output的KL散度得到的softloss太少，0.001左右
        # kl_div = F.kl_div(F.log_softmax(student_outputs, dim=1),
        #                      F.softmax(teacher_outputs / self.T, dim=1), reduction='batchmean')
        #
        
        # #  基于特征
        # cosine_similarity = F.cosine_similarity(F.normalize(student_outputs), F.normalize(teacher_outputs), dim=1)

        # 基于logits
        cosine_similarity = F.cosine_similarity(F.normalize(thetas_stu), F.normalize(thetas_tea), dim=1)
        
        soft_loss = (1-cosine_similarity).mean()

        # kd_loss=self.alpha * inter_loss + self.beta * intra_loss+self.delta*cls_loss+self.sigma*soft_loss
        kd_loss=self.alpha*cls_loss+self.beta*soft_loss #*100
        hard_loss = cls_loss
        soft_loss = soft_loss

        return kd_loss, hard_loss, soft_loss
    
    def validation_forward (self, teacher_outputs, student_outputs):
        # T_t=(teacher_outputs/self.T).softmax(dim=1)
        # T_s=(student_outputs/self.T).softmax(dim=1)
        # inter_loss = self.T**2 * inter_class_relation(T_s, T_t)
        # intra_loss = self.T**2 * intra_class_relation(T_s, T_t)
        
        # dist_loss = self.alpha * inter_loss + self.beta * intra_loss

        soft_loss = F.kl_div(F.log_softmax(student_outputs, dim=1),
                             F.softmax(teacher_outputs / self.T, dim=1), reduction='batchmean')

        return soft_loss