import torch
import torchvision.transforms as T
from models import ResIRSE,ResNet18,loss
from models.metric import ArcFace
from config import config as conf
from dataset import load_data
from models.KD_loss import KD_loss
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp
from tqdm import tqdm
from test import unique_image, group_image, featurize, compute_accuracy
from validate import validation_in_process

# Train Data Setup
dataloader, class_num = load_data(conf, training=True)
embedding_size = conf.embedding_size
device = conf.device

print(f"on device {device}")
print(f"{conf.backbone}_{conf.metric}")

# Validation Data Setup
images = unique_image(conf.test_list)
images = [osp.join(conf.test_root, img) for img in images]
groups = group_image(images, conf.test_batch_size)


# Define teacher model
teacher_model = ResIRSE(conf.embedding_size, conf.drop_ratio).to(conf.device)
teacher_model.eval()


# Define student model
student_model = ResNet18().to(conf.device)
student_model.train()

# Define metric
metric = ArcFace(embedding_size, class_num).to(device)

# Define loss function
criterion = KD_loss(class_num, T=2, alpha=0.5, beta=1.0, gamma=1.0, embedding_size=embedding_size).to(device)


# Define optimizer
optimizer = optim.SGD([{'params': student_model.parameters()}, {'params': metric.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)

# Training loop
for epoch in range(conf.epoch):
    batch_num=0
    for data, labels in tqdm(dataloader, desc=f"Epoch {epoch}/{conf.epoch}",
                             ascii=True, total=len(dataloader),mininterval=10):
        batch_num+=1

        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass with teacher model
        teacher_outputs = teacher_model(data)
        
        # Forward pass with student model
        student_outputs = student_model(data)
        
        # Compute loss -------------------------------------------------------> 修改Loss
        loss = criterion(student_outputs, teacher_outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress
        print(f"Epoch: {epoch+1}/{conf.epoch}, Loss: {loss.item()}")
        
        if batch_num % 300 == 0 :
            validation_in_process(conf, student_model, teacher_model, criterion, groups, epoch)

    # Perform validation
    if (epoch+1) % conf.validation_interval == 0:
        student_model.eval()
        
        with torch.no_grad():
            total_loss = 0
            total_samples = 0


            feature_dict_teacher = dict()
            feature_dict_student = dict()
            for group in groups:
                
                sd = featurize(group, conf.test_transform, student_model, conf.device)
                feature_dict_student.update(sd)
                td = featurize(group, conf.test_transform, teacher_model, conf.device)
                feature_dict_teacher.update(td)

                sorted_keys = sorted(feature_dict_teacher.keys())
        
                # val_teacher_outputs = feature_dict_teacher
                # val_student_outputs = feature_dict_student

                val_teacher_outputs = torch.stack([feature_dict_teacher[key] for key in sorted_keys])
                val_student_outputs = torch.stack([feature_dict_student[key] for key in sorted_keys])

                val_loss = criterion.validation_forward(val_student_outputs, val_teacher_outputs)
                total_loss += val_loss
                total_samples += len(group)
            validation_loss = total_loss / total_samples
            accuracy, threshold = compute_accuracy(feature_dict_student, conf.test_list, conf.test_root) 

            print(
                f"Validation Epoch: {epoch+1}/{conf.epoch} "
                f"Validation Loss: {validation_loss:.3f}"
                f"Accuracy: {accuracy:.3f}  "
                f"Threshold: {threshold:.3f}\n"
            )

            
    student_model.train()
    # Save the trained student model
    backbone_path = osp.join(conf.student_ckpt, f"Resnet18_{epoch}_{accuracy:.3f}_{loss:.4f}.pth")
    torch.save(student_model.state_dict(),backbone_path )