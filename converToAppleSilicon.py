# cuda上训练的模型可以直接在AppleSilicon上使用，可以使用MPS进行加速，或者直接使用CPU进行推理
# 但是需要将模型转换为CoreML格式，这样才能在iOS上使用
# 需要注意的是如果训练时使用了model = nn.DataParallel(model , device_ids=[0])
# 需要在使用之前将模型转换为普通的模型，否则会报错
# 本脚本是将模型转换为CoreML格式的脚本
# 如果需要将cuda模型部署到AppleSilicon上，可以直接使用（视具体情况将并行[DataParrale]模型转化）

import torch
import torch.nn as nn
import coremltools as ct
import torchvision
from recognition.model.resnet import ResIRSE
# Instantiate the model
model = ResIRSE(embedding_size=512, drop_ratio=0.5)
model = nn.DataParallel(model , device_ids=[0])
# Load the trained weights into the model
model.load_state_dict(torch.load('/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_arcface_56_3.3647572994232178.pth', map_location=torch.device('cuda:0')))
#model.load_state_dict(torch.load('/home/zjb/workbench/checkpoints/ckpt-recognition/Tested/resnet_arcface_56_3.3647572994232178.pth'))
# Set the model to evaluation mode
model.eval()
# Create an example input
example_input = torch.rand(1, 1, 128, 128)
# Trace the model using the instantiated object
traced_model = torch.jit.trace(model, example_input)
out = traced_model(example_input)
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

image_input = ct.ImageType(name="input_1",
                           shape=example_input.shape,
                           scale=scale, bias=bias)

# # Convert the traced model to CoreML
# coreml_model = ct.convert(traced_model,
#                           convert_to="mlprogram",
#                           inputs=[ct.TensorType(name="input", 
#                                                 shape=example_input.shape)])
# # Save the CoreML model
# coreml_model.save('/path/to/save/model.mlmodel')

model = ct.convert(
    traced_model,
    inputs=[image_input],
    #classifier_config = ct.ClassifierConfig(class_labels),
    compute_units=ct.ComputeUnit.CPU_ONLY,
)