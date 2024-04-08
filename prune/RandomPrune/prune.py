# -*- coding: utf-8 -*-

'''Deep Compression with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp

import argparse

from model import *
# from utils import progress_bar

import numpy as np

from config import config as conf

from dataset import load_data
from model import FocalLoss , ArcFace
from test_diff_dataset import test_in_train
import time


parser = argparse.ArgumentParser(description='PyTorch ResNet18 Pruning')
parser.add_argument('--loadfile', '-l', default="/home/zjb/workbench/checkpoints/ckpt-KD/_record_Resnet18_29_0.953_3.6503.pth",dest='loadfile')
parser.add_argument('--prune', '-p', default=0.25, dest='prune', help='Parameters to be pruned')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--net', default='res18')
args = parser.parse_args()

prune = float(args.prune)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
conf.device = device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
dataloader, class_num = load_data(conf, training=True)

# Model
print('==> Building model..')
if args.net=='res18':
    net = ResNet18()
else :
    print(f"model {args.net} not found!")
    
net = net.to(device)



# Load weights from checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isfile(args.loadfile), 'Error: no checkpoint directory found!'
checkpoint = {}
checkpoint['net'] = torch.load(args.loadfile)
net.load_state_dict(checkpoint['net'])
acc, th = test_in_train(net, conf.test_list, conf.test_root, conf)
print(f"first prepruned acc:",acc)
    
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def prune_weights(torchweights):
    weights=np.abs(torchweights.cpu().numpy())
    weightshape=weights.shape
    rankedweights=weights.reshape(weights.size).argsort()#.reshape(weightshape)
    
    num = weights.size
    prune_num = int(np.round(num*prune))
    count=0
    masks = np.zeros_like(rankedweights)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n]=1
        else: count+=1
    print("total weights:", num)
    print("weights pruned:",count)
    
    masks=masks.reshape(weightshape)
    weights=masks*weights
    
    return torch.from_numpy(weights).cuda(), masks
    
# prune weights
# The pruned weight location is saved in the addressbook and maskbook.
# These will be used during training to keep the weights zero.
addressbook=[]
maskbook=[]
for k, v in net.state_dict().items():
    if "conv2" in k:
        addressbook.append(k)
        print("pruning layer:",k)
        weights=v
        weights, masks = prune_weights(weights)
        maskbook.append(masks)
        checkpoint['net'][k] = weights
        
checkpoint['address'] = addressbook
checkpoint['mask'] = maskbook
net.load_state_dict(checkpoint['net'])

# Training

metric_f = ArcFace(embedding_size=conf.embedding_size, class_num=class_num).to(device)
checkpoint_arcface = torch.load('/home/zjb/workbench/checkpoints/ckpt-recognition/arcface_weight/arcface_resnet18_26_2.7453722953796387.pth')
metric_f.load_state_dict(checkpoint_arcface,strict=False)
# metric.load_state_dict(torch.load("/home/zjb/workbench/checkpoints/ckpt-recognition/arcface_weight/arcface_resnet18_30_4.202079772949219.pth"))

criterion = FocalLoss(gamma=2)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric_f.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    e=epoch
    net.train()
    step = 50
    temp_loss = 0.0
    batch_count = 0
    avg_loss = 0.0
    net.train()
    for data, labels in dataloader :
        batch_count += 1
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = net(data)
        thetas = metric_f(embeddings, labels)
        loss = criterion(thetas, labels)
        loss.backward()
        optimizer.step()

        
        # mask pruned weights 
        checkpoint['net']=net.state_dict()
#        print("zeroing..")
#        print(np.count_nonzero(checkpoint['net'][addressbook[0]].cpu().numpy()))
        for address, mask in zip(addressbook, maskbook):
#            print(address)
            checkpoint['net'][address] = torch.from_numpy(checkpoint['net'][address].cpu().numpy() * mask)
#        print(np.count_nonzero(checkpoint['net'][addressbook[0]].cpu().numpy()))  

        temp_loss = loss.item()
        avg_loss += temp_loss
        if batch_count % step == 0:
            net.eval()
            acc,th=test_in_train(net, conf.test_list, conf.test_root, conf)
            print(f"prune_resnet18\t{e+1}\t\t{batch_count}\t\t{len(dataloader)}\t\t{acc:.5f}\t\t{th:.5f}\t\t{avg_loss/step:.6f}\t{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            avg_loss = 0
            net.train()

for epoch in range(start_epoch, start_epoch+250):

    acc,th=test_in_train(net, conf.test_list, conf.test_root, conf)
    print(f"prepruned acc:",acc)

    train(epoch)

    acc,th=test_in_train(net, conf.test_list, conf.test_root, conf)
    print(f"Epoch:{epoch} pruned acc:",acc)      

    backbone_path = osp.join("checkpoints/ckpt-prune/random-prune", f"ResNet18_pruned_{conf.metric}_{epoch}_{acc:.3f}.pth")
    torch.save(net.state_dict(), backbone_path)
    scheduler.step()