import torch

import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class BasicBlock_with_config(nn.Module):
    expansion = 1

    def __init__(self, in_channels, mid_channels,out_channels, stride=1):
        super(BasicBlock_with_config, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet18_with_config(nn.Module):
    # cfg= [512, 19, 'M', 64, 63, 64, 63, 128, 121, 2, 128, 120, 191, 248, 0, 95, 121, 40, 57, 0, 175, 444]
    def __init__(self, block=BasicBlock_with_config, layers=[2,2,2,2], embedding_size=512,config=None):
        super(ResNet18_with_config, self).__init__()
    
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(512 * 8 * 8, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.in_channels = config[1]
        self.conv1 = nn.Conv2d(1, config[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(config[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], config[3:7], stride=1)
        self.layer2 = self._make_layer(block, layers[1], config[7:12], stride=2)
        self.layer3 = self._make_layer(block,  layers[2], config[12:17], stride=2)
        self.layer4 = self._make_layer(block,layers[3], config[17:22], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config[-2] * block.expansion, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size) 

    def _make_layer(self, block, blocks, cfg, stride=1):
        layers = []
        layers.append(block(self.in_channels, cfg[0],cfg[1], stride))
        self.in_channels = cfg[1] * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, cfg[2] ,cfg[3]))
            self.in_channels = cfg[3] * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)                                   
        x = self.bn(x)                                       ##################   

        return x


if __name__ == "__main__":
    model = ResNet18_with_config(config= [512, 19, 'M', 64, 63, 64, 63, 128, 121, 2, 128, 120, 191, 248, 23, 95, 121, 40, 57, 65, 175, 444])
    print(1)
    print(model)
    x =  torch.randn(64,1,128,128)
    print(model(x).shape)