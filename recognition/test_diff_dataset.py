# -*- coding: utf-8 -*-
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from config import config as conf
from model import FaceMobileNet
from model import ResIRSE
from model import ResNet18

import onnx

import sys

def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i : end])
    return res


def _preprocess(images: list, transform) -> torch.Tensor:
    res = []
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
    data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    data = data[:, None, :, :]    # shape: (batch, 1, 128, 128)
    return data


def featurize(images: list, transform, net, device) -> dict:
    """featurize each image and save into a dictionary
    Args:
        images: image paths
        transform: test transform
        net: pretrained model
        device: cpu or cuda
    Returns:
        Dict (key: imagePath, value: feature)
    """
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data) 
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def compute_accuracy(feature_dict, pair_list, test_root):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

def test(model,pth_path,testList,testRoot):
    if model == "resnet18" :
        model = ResNet18().to(conf.device)
    elif model == "resnet50" :
        model = ResIRSE(conf.embedding_size, conf.drop_ratio).to(conf.device)
    model = nn.DataParallel(model , device_ids=conf.deviceID)
    model.load_state_dict(torch.load(conf.test_model, map_location=conf.device))

    model.eval()

    images = unique_image(testList)
    images = [osp.join(testRoot, img) for img in images]
    groups = group_image(images, conf.test_batch_size)

    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, testList, testRoot) 

    testDataSet=testRoot.split("/")[-1]

    print(
        f"Test Model: {conf.test_model}\n"
        f"Test DataSet {testDataSet}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )


def test_in_train(model,testList,testRoot,conf):

    model.eval()
    images = unique_image(testList)
    images = [osp.join(testRoot, img) for img in images]
    groups = group_image(images, conf.test_batch_size)
    feature_dict = dict()
    for group in groups:
        d = featurize(group, conf.test_transform, model, conf.device)
        feature_dict.update(d) 
    accuracy, threshold = compute_accuracy(feature_dict, testList, testRoot) 
    # testDataSet=testRoot.split("/")[-1]
    # print(
    #     f"Test Model: {conf.test_model}\n"
    #     f"Test DataSet {testDataSet}\n"
    #     f"Accuracy: {accuracy:.3f}\n"
    #     f"Threshold: {threshold:.3f}\n"
    # )
    return accuracy,threshold

if __name__ == '__main__':

    if len(sys.argv)==2 :
        conf.test_model = f"../checkpoints/ckpt-recognition/{sys.argv[1]}.pth"
    #model = FaceMobileNet(conf.embedding_size)

    conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_arcface_56_3.3647572994232178.pth"
    conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/resnet18_arcface_49_2.596092462539673.pth"
    print(f"Test Model: {conf.test_model}")
    model="resnet18" # or resnet18

    # AgeDB
    test(model=model,pth_path=conf.test_model,testList=conf.age_test_list,testRoot=conf.age_test_root)
        
    # LFW
    test(model=model,pth_path=conf.test_model,testList=conf.test_list,testRoot=conf.test_root)

    # Train set
    test(model=model,pth_path=conf.test_model,testList=conf.test_on_train_list,testRoot=conf.test_on_train_root)

