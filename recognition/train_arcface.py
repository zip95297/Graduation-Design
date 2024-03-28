# 单独训练在CASIA-WebFace数据集上的arcface中的权重

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import FaceMobileNet, ResIRSE
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf

# Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

print(f"on device {device}")
print(f"{conf.backbone}_{conf.metric}")

# model = ResIRSE(conf.embedding_size, conf.drop_ratio)
# # 没有这个就不能加载模型
# # model = nn.DataParallel(model)
# model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))
# model.eval()


# Network Setup
if conf.backbone == 'resnet':
    net = ResIRSE(embedding_size, conf.drop_ratio).to(device)
    net = nn.DataParallel(net,device_ids=conf.deviceID)
    net.load_state_dict(torch.load(conf.test_model,map_location=device))
else:
    net = FaceMobileNet(embedding_size).to(device)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
else:
    metric = CosFace(embedding_size, class_num).to(device)


# Training Setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    # optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
    #                         lr=conf.lr, weight_decay=conf.weight_decay)
    optimizer = optim.SGD([{'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
else:
    # optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
    #                         lr=conf.lr, weight_decay=conf.weight_decay)
    optimizer = optim.Adam([{'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

# Checkpoints Setup
checkpoints = conf.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if conf.restore:
    weights_path = osp.join(checkpoints, conf.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))

# Start training
# net.train()
net.eval()

for e in range(conf.epoch):
    total_batch_num = len(dataloader)
    batch_count = 0
    for data, labels in dataloader:
        batch_count += 1
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()
        if batch_count % 25 == 0:
            print(f"Epoch {e}/{conf.epoch}, batch {batch_count}/{total_batch_num}, Loss: {loss}")

    metric_path = osp.join(checkpoints,"arcface_weight",f"{conf.metric}_{e}_{loss}.pth")
    torch.save(metric.state_dict(), metric_path)
    scheduler.step()