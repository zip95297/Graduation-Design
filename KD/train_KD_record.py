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
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from test import unique_image, group_image
from validate import validation_in_process, validation_teacher_model

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
teacher_model = nn.DataParallel(teacher_model,device_ids=conf.deviceID)
teacher_model.load_state_dict(torch.load(conf.teacher_model_path,map_location=device))
teacher_model.eval()


# Define student model
student_model = ResNet18().to(conf.device)
student_model.train()

# Define metric
# metric = ArcFace(embedding_size, class_num).to(device)

# Define loss function
criterion = KD_loss(class_num, T=2, alpha=0.5, beta=1.0, gamma=1.0, embedding_size=embedding_size).to(device)


# Define optimizer
if conf.KD_optimizer == 'sgd':
    optimizer = optim.SGD([{'params': student_model.parameters()}, {'params': criterion.parameters()}], 
                            lr=conf.lr, weight_decay=conf.weight_decay)      #metric originally
else:
    optimizer = optim.Adam([{'params': student_model.parameters()}, {'params': criterion.parameters()}],
                            lr=conf.lr, weight_decay=conf.weight_decay)

#------------------------------------------------------------------------------------>scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=conf.lr_step, gamma=0.1)

print(f"Training {conf.backbone} with {conf.metric} with {conf.epoch} epochs and {conf.train_batch_size} batch size and optimizer {conf.KD_optimizer}")


teacher_model.eval()
student_model.train()
# 评估teacher model
print("Evaluating teacher model...")
validation_teacher_model(teacher_model, conf, groups)

print("start training...")
# Training loop
for epoch in range(conf.epoch):
    teacher_model.eval()
    student_model.train()
    batch_num=0
    batch_num_total = len(dataloader)
    avg_loss = 0
    
    for data, labels in dataloader:
            
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
        # 限制梯度范数 防止梯度爆炸
        clip_grad_norm_(student_model.parameters(), max_norm=5, norm_type=2)
        clip_grad_norm_(criterion.parameters(), max_norm=5, norm_type=2)
        
        optimizer.step()
        
        
        # # compute gradients--------------------------------------------------------------x waited to be deleted
        # for p in student_model.parameters():
        #     print(p.requires_grad)
        # gradients = [p.grad.norm().item() for p in student_model.parameters()]
        
        # print(f'Epoch {epoch+1}, Gradients: {gradients}')
        
        # Perform validation
        if batch_num % 100 == 0 :
            validation_in_process(conf, student_model, teacher_model, criterion, groups, epoch)

        # Print training progress
        avg_loss+=loss.item()

        show_step = 25
        if batch_num % show_step == 0:
            print(f"Epoch: {epoch+1}/{conf.epoch}, Loss: {avg_loss/show_step:.8f} ,\tbatch: {batch_num}/{batch_num_total}")
            avg_loss = 0

        

    # Perform validation
    # if (epoch+1) % conf.validation_interval == 0:
    acc,th=validation_in_process(conf, student_model, teacher_model, criterion, groups, epoch)
    print(f"Epoch: {epoch+1}/{conf.epoch}, Accuracy: {acc:.5f}, Threshold: {th:.5f}")

    student_model.train()
    scheduler.step() #------------------------------------------------------------->scheduler
    # Save the trained student model
    backbone_path = osp.join(conf.student_ckpt, f"Resnet18_{epoch}_{acc:.3f}_{loss:.4f}.pth")
    torch.save(student_model.state_dict(),backbone_path )