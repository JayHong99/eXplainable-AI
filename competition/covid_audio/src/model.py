
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

class Conv2d_Block(nn.Module) : 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) : 
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels = in_channels, 
                                    out_channels = out_channels, 
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding,
                                    )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        init.kaiming_normal_(self.conv_layer.weight, a = 0.1)
        self.conv_layer.bias.data.zero_()

    def forward(self, x) : 
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        return self.relu(x)



class AudioClassifier (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d_Block( 2,  8, (5,5), (2,2), (2,2))
        self.conv2 = Conv2d_Block( 8, 16, (3,3), (2,2), (1,1))
        self.conv3 = Conv2d_Block(16, 32, (3,3), (2,2), (1,1))
        self.conv4 = Conv2d_Block(32, 64, (3,3), (2,2), (1,1))

        self.conv = nn.Sequential(*[self.conv1, self.conv2, self.conv3, self.conv4])
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)

        self.mlp = nn.Linear(64, 2)
 
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.pooling(x)
        x = x.view(batch_size, -1)
        return self.mlp(x)