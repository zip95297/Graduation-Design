import torch
import torch.nn as nn
import os
import numpy as np
from models.resnet18 import ResNet18

ckpt="/home/zjb/workbench/checkpoints/ckpt-KD/Sparsify/_record_Resnet18_Sparsify_18_0.952_3.4756.pth"

percent = 0.9  # scale sparse rate

model = ResNet18()
model.load_state_dict(torch.load(ckpt))

kernnels = []
pre = ""
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) and isinstance(pre, nn.Conv2d):
        print(pre)
        print(m.weight.data.shape[0])
        kernnels.append(m.weight.data.shape[0])
    pre = m

print(len(kernnels))

gammas = []

pre = ""
index=0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) and isinstance(pre, nn.Conv2d):
        index+=1
        s=[gammas.append((index,v)) for v in m.weight.data.numpy().tolist()]
    pre = m

# print(gammas)

print(kernnels)

gammas = sorted(gammas, key=lambda x: abs(x[1]))

count = len(gammas)

to_pruned = gammas[:int(count*percent)]

for i in to_pruned:
    kernnels[i[0]-1]-=1

print(kernnels)
# print(gammas)