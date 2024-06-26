# -*- coding: utf-8 -*-
import os
import os.path as osp

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from recognition.config import config as conf
from recognition.model import FaceMobileNet
from recognition.model import ResIRSE
from recognition.model import ResNet18, ResNet18_with_config

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

def test(model,pth_path,testList,testRoot,cfg=None,choice=None):
    if model == "resnet18" :
        model = ResNet18().to(conf.device)
    elif model == "resnet50" :
        model = ResIRSE(conf.embedding_size, conf.drop_ratio).to(conf.device)
    elif model == "resnet18_with_cfg" :
        model = ResNet18_with_config(config= cfg).to(conf.device)

    if choice == "PearFace" or choice == "ArcFace" or choice == "CosFace" or choice == "resnet18_alone":
        model = nn.DataParallel(model , device_ids=conf.deviceID)
    model.load_state_dict(torch.load(pth_path, map_location=conf.device))

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
        f"Accuracy: {accuracy*100:.3f}%\n"
        f"Threshold: {threshold:.10f}\n"
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


                        # if len(sys.argv)==2 :
                        #     conf.test_model = f"../checkpoints/ckpt-recognition/{sys.argv[1]}.pth"
                        # #model = FaceMobileNet(conf.embedding_size)

                        # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_arcface_56_3.3647572994232178.pth"
                        # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/resnet18_arcface_49_2.596092462539673.pth"
                        # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/channel-prune/_record_Resnet18_pruned_35_0.949_4.2501.pth"
                        



                        # cfg=None
                        # cfg=[512, 55, 'M', 64, 64, 64, 63, 128, 128, 53, 128, 128, 256, 256, 2, 256, 222, 512, 507, 1, 442, 510]
                        # cfg=[512, 60, 'M', 64, 64, 64, 63, 128, 128, 53, 128, 128, 256, 256, 8, 247, 222, 497, 377, 5, 420, 510]
                        
                        # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/channel-prune/_record_Resnet18_pruned_30_0.950_3.3877.pth"
                        # model="resnet18_with_cfg" # or resnet18

                        # # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/Sparsify/_record_Resnet18_Sparsify_33_0.955_4.7505.pth"
                        # # model="resnet18"

                        # # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/_record_Resnet18_29_0.953_3.6503.pth"
                        # # model="resnet18"
                        
                        # # conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/random-prune/ResNet18_pruned_arcface_64_0.862.pth"



#if __name__ == '__main__':


    # model in :
    #   [
    #     resnet18_KD_feature,
    #     resnet18_KD_logits,
    #     resnet18_with_cfg_pruned_channel,
    #     resnet18_with_cfg_pruned_random,
    #     resnet18_KD_sparsity,
    #     resnet18_alone,
    #     PearFace,
    #     ArcFace,
    #     CosFace
    #   ] 
def test_diff_dataset(choice_id=0):

    choices = [
        "resnet18_with_cfg_pruned_channel",
        "resnet18_with_cfg_pruned_random",
        "resnet18_KD_sparsity",
        "resnet18_alone",
        "resnet18_KD_feature",
        "resnet18_KD_logits",
        "PearFace",
        "ArcFace",
        "CosFace"
    ]

    cfg=None

    choice=choices[choice_id]

    if choice == "resnet18_KD_feature" :
        model = "resnet18"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/_record_Resnet18_29_0.953_3.6503.pth"

    if choice == "resnet18_KD_logits" :
        model = "resnet18"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/Resnet18_89_0.947_2.5306.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/Resnet18_88_0.948_3.5106.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/Resnet18_90_0.946_4.9533.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/StudentWithoutDataParalle.pth"

    if choice == "resnet18_with_cfg_pruned_channel" :
        model="resnet18_with_cfg"
        cfg=[512, 60, 'M', 64, 64, 64, 63, 128, 128, 53, 128, 128, 256, 256, 8, 247, 222, 497, 377, 5, 420, 510]
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/channel-prune/_record_Resnet18_pruned_35_0.949_4.2501.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/channel-prune/_record_Resnet18_pruned_30_0.950_3.3877.pth"

    if choice == "resnet18_with_cfg_pruned_random" :
        model="resnet18"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-prune/random-prune/ResNet18_pruned_arcface_64_0.862.pth"
               
    
    if choice == "resnet18_KD_sparsity" :
        model = "resnet18"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-KD/Sparsify/_record_Resnet18_Sparsify_33_0.955_4.7505.pth"

    if choice == "resnet18_alone" :
        model = "resnet18"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/resnet18_pearface_49_2.596092462539673.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/resnet18_pearface_48_5.69857931137085.pth"

    if choice == "PearFace" :
        model = "resnet50"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/TeacherModel/teacher_now.pth"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_pearface_56_3.3647572994232178.pth"
    
    if choice == "ArcFace" :
        model = "resnet50"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_arcface_52_4.424378871917725.pth"
        pass
    
    if choice == "CosFace" :
        model = "resnet50"
        conf.test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/resnet_cosface_39_4.565798282623291.pth"
        pass
        
    print(f"Model: {choice}")
    print(f"Network: {model}")
    print(f"Test CheckPoint: {conf.test_model}")
   
    # LFW
    test(model=model,pth_path=conf.test_model,testList=conf.test_list,testRoot=conf.test_root,cfg=cfg,choice=choice)

    # AgeDB
    test(model=model,pth_path=conf.test_model,testList=conf.age_test_list,testRoot=conf.age_test_root,cfg=cfg,choice=choice)
        
    # Train set
    test(model=model,pth_path=conf.test_model,testList=conf.test_on_train_list,testRoot=conf.test_on_train_root,cfg=cfg,choice=choice)

if __name__ == '__main__':

    choices = [
        "resnet18_with_cfg_pruned_channel",
        "resnet18_with_cfg_pruned_random",
        "resnet18_KD_sparsity",
        "resnet18_alone",
        "resnet18_KD_feature",
        "resnet18_KD_logits",
        "PearFace",
        "ArcFace",
        "CosFace"
    ]

    test_diff_dataset(-2)

    for i in range(9):
        test_diff_dataset(i)