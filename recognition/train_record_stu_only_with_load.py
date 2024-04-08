import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import FaceMobileNet, ResIRSE, ResNet18
from model.metric import ArcFace, CosFace
from model.loss import FocalLoss
from dataset import load_data
from config import config as conf
from test_diff_dataset import test_in_train
import time

# Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

# # ----------------------------------------------------------------------------------->直接在这里改显卡
# conf.deviceID=[2]
# device = f'cuda:{conf.deviceID[0]}' if torch.cuda.is_available() else 'cpu'

conf.backbone = 'resnet18' 

print(f"on device {device}")
print(f"{conf.backbone}_{conf.metric}")

# Network Setup
if conf.backbone == 'resnet':
    net = ResIRSE(embedding_size, conf.drop_ratio).to(device)
elif conf.backbone == 'resnet18':
    net = ResNet18().to(device)
else:
    net = FaceMobileNet(embedding_size).to(device)

#--------------------------------------------------------------------------------------------------> load
checkpoint = {}
checkpoint['net'] = torch.load("/home/zjb/workbench/checkpoints/ckpt-KD/_record_Resnet18_29_0.953_3.6503.pth")
net.load_state_dict(checkpoint['net'])
acc, th = test_in_train(net, conf.test_list, conf.test_root, conf)

if conf.metric == 'arcface':
    metric = ArcFace(embedding_size, class_num).to(device)
else:
    metric = CosFace(embedding_size, class_num).to(device)

checkpoint_arcface = torch.load('/home/zjb/workbench/checkpoints/ckpt-recognition/arcface_weight/arcface_resnet18_26_2.7453722953796387.pth')
metric.load_state_dict(checkpoint_arcface,strict=False)

net = nn.DataParallel(net, device_ids=conf.deviceID)
metric = nn.DataParallel(metric, device_ids=conf.deviceID)

# Training Setup
if conf.loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = nn.CrossEntropyLoss()

if conf.optimizer == 'sgd':
    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)
else:
    optimizer = optim.Adam([{'params': net.parameters()}, {'params': metric.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

# Checkpoints Setup
checkpoints = conf.checkpoints
os.makedirs(checkpoints, exist_ok=True)

if conf.restore:
    weights_path = osp.join(checkpoints, conf.restore_model)
    net.load_state_dict(torch.load(weights_path, map_location=device))

# Start training
net.train()

print("start training...")
print("model\t\tepoch\tbatch\ttotalbatch\taccuaracy\tthreshold\tloss\t\ttime")

step=1

for e in range(conf.epoch):
    temp_loss=0

    # # 设置迭代器和进度条
    # bar = tqdm(dataloader, desc=f"Epoch {e}/{conf.epoch} Loss:{temp_loss:.4f}",
    #                          ascii=True, total=len(dataloader),mininterval=2)
    batch_count = 0
    avg_loss = 0
    for data, labels in dataloader :
        batch_count += 1
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()
        temp_loss=loss.item()
        avg_loss += temp_loss

        if batch_count % step == 0:
            net.eval()
            acc,th=test_in_train(net, conf.test_list, conf.test_root, conf)
            print(f"{conf.backbone}\t{e+1}\t\t{batch_count}\t\t{len(dataloader)}\t\t{acc:.5f}\t\t{th:.5f}\t\t{avg_loss/step:.6f}\t{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            avg_loss = 0
            net.train()

        
        # # 实时显示loss
        # bar.set_description(f"Epoch {e}/{conf.epoch} Loss:{temp_loss:.4f}")

    # print(f"Epoch {e}/{conf.epoch}, Loss: {loss}")

    backbone_path = osp.join(checkpoints, f"{conf.backbone}_{conf.metric}_{e}_{loss}.pth")
    torch.save(net.state_dict(), backbone_path)
    scheduler.step()