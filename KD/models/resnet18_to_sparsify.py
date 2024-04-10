import torch

import torch.nn as nn
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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


class ResNet18_to_Sparsify(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2,2], embedding_size=512,sparsify_gamma=0.0001):
        super(ResNet18_to_Sparsify, self).__init__()

        self.sparsify_gamma = sparsify_gamma
        
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(512 * 8 * 8, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.grad is not None:   #   实际运行发现outputlayer的梯度为None为什么？ 梯度消失了 为什么会这样
                    m.weight.grad.data.add_(self.sparsify_gamma*torch.sign(m.weight.data))  # L1

    def updateBN_without_res(self):
        last=nn.Conv2d(1,1,kernel_size=3)
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d) and isinstance(last,nn.Conv2d) and last.kernel_size!=(1,1): # and last.kernel_size!=(7,7)  这样只包含了block中的
                #print(last.kernel_size)
                m.weight.grad.data.add_(self.sparsify_gamma*torch.sign(m.weight.data))  # L1
            last=m

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
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
    from PIL import Image
    import numpy as np

    model = ResNet18_to_Sparsify()


    x = torch.randn(64, 1, 128, 128)
    net = ResNet18_to_Sparsify(embedding_size=512)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=0.0001)
    
    
    net.train()
    while True:

        res = net(x)
        loss = res.sum()*100000
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                print(m.weight.grad)

        optimizer.step()
        # model.updateBN()

    net.eval()
    with torch.no_grad():
        out = net(x)
    print(out.shape)