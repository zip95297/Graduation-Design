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