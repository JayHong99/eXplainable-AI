from turtle import forward
import torch
import torch.nn as nn
from torch.nn.functional import relu

class ConvBlock(nn.Module) : 
    def __init__(self, in_channel, out_channel, max_pool_size) :
        super(ConvBlock, self).__init__()

        self.conv_layer1 = nn.Conv2d(in_channel, out_channel, 
                                    kernel_size = 3, 
                                    stride = 1, 
                                    padding = 1, 
                                    bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.drop = nn.Dropout(0.25, inplace=True)

        self.conv_layer2 = nn.Conv2d(out_channel, out_channel, 
                                    kernel_size = 3, 
                                    stride = 1, 
                                    padding = 1, 
                                    bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = nn.MaxPool2d(2, stride = 1)
        self.max_pool_size = max_pool_size

    def forward(self, x) : 
        out = self.conv_layer1(x)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.relu(out)

        out = self.conv_layer2(out)
        out = self.bn1(out)
        out = self.drop(out)
        out = self.relu(out)

        return nn.MaxPool2d(self.max_pool_size)(out)

class CNN_Model(nn.Module) : 
    def __init__(self, block, num_classes) : 
        super(CNN_Model, self).__init__()
        self.layer1 = block(3,16,2)
        self.layer2 = block(16,32,2)
        self.layer3 = block(32,64,2)
        self.layer4 = block(64,128,7)
        self.classifier = nn.Sequential(
                                nn.Linear(2048, num_classes)
                                        )
    def forward(self, x) : 
        batch_size = x.size(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

def CNN_model(num_classes) : 
    return CNN_Model(ConvBlock, num_classes)
    
    