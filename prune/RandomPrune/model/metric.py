# Definition of ArcFace loss and CosFace loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):

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
            s: scale, see normface https://arxiv.org/abs/1704.06369
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
#         input_mean = torch.mean(x, dim=1, keepdim=True)
#         weight_mean = torch.mean(self.weight, dim=1, keepdim=True)
        
#         # 计算输入和权重减去均值后的内积
#         input_centered = x - input_mean
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
        # cosine 该为皮尔逊相关性系数
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
        # 计算输入和权重的均值
        x= F.normalize(x)
        weight=self.weight / self.weight.norm(dim=1, keepdim=True)


        input_mean = torch.mean(x, dim=1, keepdim=True)
        ## 1->0
        weight_mean = torch.mean(weight, dim=0, keepdim=True)
        
        # 计算输入和权重减去均值后的内积
        #input_centered = x - input_mean
        input_centered = x - weight_mean
        weight_centered = self.weight - weight_mean
        
        # # normalize?
        # inner_product = torch.mm(input_centered, weight_centered.T)
        
        # # 计算对应的标准差
        # input_std = torch.std(input_centered, dim=1, unbiased=False, keepdim=True)
        # weight_std = torch.std(weight_centered, dim=1, unbiased=False, keepdim=True)
        
        # # 计算皮尔逊相关系数的近似值
        # pearson_coeffs = inner_product / (torch.mm(input_std, weight_std.T) + 1e-8)

        pearson=F.linear(F.normalize(input_centered), F.normalize(weight_centered))

        return pearson
    
if __name__ == "__main__":

    checkpoint = torch.load('/home/zjb/workbench/checkpoints/ckpt-recognition/arcface_weight/arcface_resnet18_26_2.7453722953796387.pth')

    # 实例化ArcFace模型
    embedding_size = 512  # 例如
    class_num = 10575       # 例如
    model = ArcFace(embedding_size, class_num)

    # 检查.pth文件中的键，确保它包含'weight'或对应的键
    print(checkpoint.keys())

    # 如果.pth文件直接包含了权重矩阵，则可以直接加载
    if 'weight' in checkpoint:
        model.weight = nn.Parameter(checkpoint['weight'])
    else:
        # 或者，如果.pth文件包含了一个状态字典
        # 比如，如果状态字典的键是'module.weight'或其他，请相应调整
        model.load_state_dict(checkpoint, strict=False)

    p=PearFace(512, 10575)
    x=torch.randn(64,512)
    label=torch.randint(0,10575,(64,))
    output=p(x,label)