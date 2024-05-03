# Definition of ArcFace loss and CosFace loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        """ArcFace formula: 
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether 
        (m + theta) go out of [0, Pi]

        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s


class CosFace(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        """
        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
    
# class PearFace(nn.Module):
#     def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
#         """PearFace formula: 
#             theta = uncentered(theta)
#             cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
#         Note that:
#             0 <= m + theta <= Pi
#         So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
#         we have:
#             cos(theta) < cos(Pi - m)
#         So we can use cos(Pi - m) as threshold to check whether 
#         (m + theta) go out of [0, Pi]

#         Args:
#             embedding_size: usually 128, 256, 512 ...
#             class_num: num of people when training
#             s: scale
#             m: margin
#         """
#         super().__init__()
#         self.in_features = embedding_size
#         self.out_features = class_num
#         self.s = s
#         self.m = m
#         self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
#         nn.init.xavier_uniform_(self.weight)

#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, input, label):
#         # cosine 该为皮尔逊相关性系数
#         cosine = self.pearson(input)
        
#         sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
#         phi = cosine * self.cos_m - sine * self.sin_m
#         phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
#         # update y_i by phi in cosine
#         output = cosine * 1.0  # make backward works
#         batch_size = len(output)
#         output[range(batch_size), label] = phi[range(batch_size), label]
#         return output * self.s
    
#     # 计算 pearson 相关性系数 ，embeding 与 weight 的内积
#     def pearson(self, x):
#         # 计算输入和权重的均值
#         x= F.normalize(x)
#         weight=self.weight / self.weight.norm(dim=1, keepdim=True)


#         input_mean = torch.mean(x, dim=1, keepdim=True)
#         ## 1->0
#         weight_mean = torch.mean(weight, dim=0, keepdim=True)
        
#         # 计算输入和权重减去均值后的内积
#         #input_centered = x - input_mean
#         input_centered = x - weight_mean
#         weight_centered = self.weight - weight_mean
        
#         # # normalize?
#         # inner_product = torch.mm(input_centered, weight_centered.T)
        
#         # # 计算对应的标准差
#         # input_std = torch.std(input_centered, dim=1, unbiased=False, keepdim=True)
#         # weight_std = torch.std(weight_centered, dim=1, unbiased=False, keepdim=True)
        
#         # # 计算皮尔逊相关系数的近似值
#         # pearson_coeffs = inner_product / (torch.mm(input_std, weight_std.T) + 1e-8)

#         pearson=F.linear(F.normalize(input_centered), F.normalize(weight_centered))

#         return pearson
        
        

class PearFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        """PearFace formula: 
            theta = uncentered(theta)
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether 
        (m + theta) go out of [0, Pi]

        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale
            m: margin
        """
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # cosine 即为去中心化后的特征向量和权重向量的夹角余弦值
        cosine = self.pearson(input)
        
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
    
    # 计算 pearson 相关性系数 ，embeding 与 weight 的内积
    def pearson(self, x):
        # 将输入向量单位化（归一化），使每个样本的特征向量长度变为1
        x= F.normalize(x)

        # 将每个类别向量单位化（归一化），使其长度变为1
        weight=self.weight/self.weight.norm(dim=1, keepdim=True)
        
        # 计算所有权重向量的均值，即所有类别的中心向量，以进行去中心化操作
        weight_mean = torch.mean(weight, dim=0, keepdim=True)
        
        # 计算输入和权重减去均值后的内积
        input_uncentered = x - weight_mean
        weight_uncentered = self.weight - weight_mean

        # 计算处理后的输入和权重的余弦值，即thetas
        cosine_thetas = F.linear(F.normalize(input_uncentered), F.normalize(weight_uncentered))

        return cosine_thetas
        
        
if __name__ == "__main__" :
    metric = PearFace(512, 10)
    input = torch.rand(64, 512)
    label = torch.randint(0, 10, (64,))
    logits = metric(input, label)
    print(logits)
