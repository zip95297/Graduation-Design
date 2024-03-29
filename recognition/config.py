import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'resnet' # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5

    # data preprocess
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # dataset
    train_root = '/home/zjb/workbench/data/CASIA-WebFace'
    test_root = "/home/zjb/workbench/data/lfw-align-128"
    test_list = "/home/zjb/workbench/data/lfw_test_pair.txt"

    age_test_root="/home/zjb/workbench/data/AgeDB"
    age_test_list="/home/zjb/workbench/data/AgeDB_test_pair.txt"
    
    # training settings
    checkpoints = "../checkpoints/ckpt-recognition"
    restore = False
    restore_model = ""
    name="teacher_now"
    test_model = f"/home/zjb/workbench/checkpoints/ckpt-recognition/TeacherModel/{name}.pth"
    
    train_batch_size = 64
    test_batch_size = 60

    epoch = 250
    optimizer = 'sgd'  # ['sgd', 'adam']
    lr = 1e-1
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
   
    # nvidia-smi
    deviceID=[2]
    
    device = f'cuda:{deviceID[0]}' if torch.cuda.is_available() else 'cpu'

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 4  # dataloader

    # Knowlege Distilling
    teacher_model="resnet"
    student_model="resnet18"
    teacher_model_path="/home/zjb/workbench/KD/teacher/resnet_arcface_56_3.3647572994232178.pth"

config = Config()
