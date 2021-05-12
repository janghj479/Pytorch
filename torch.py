import torch
import torch.nn as nn
import torch.nn.functional as F


"""
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
"""


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
                
        # Layer 1
        self.conv2D = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), padding_mode = "reflect")
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Layer 2
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(8,1), 
                                                                          padding_mode ="replicate", groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        
        # Layer 3
        self.pointwise = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d(1, 8)
        
        # FC Layer
        self.fc1 = nn.Linear(1*64*128,16*1*4)
        

    def forward(self, x):
    
        print("input",x.size( ))
        # Layer 1
        x = self.conv2D(x)
        print("conv2D",x.size( ))
        x = self.batchnorm1(x)
        print("batchnorm",x.size( ))
        # Layer 2
        x = self.depthwise(x)
        print("depthwise",x.size( ))
        x = F.elu(self.batchnorm2(x))
        print("batchnorm",x.size( ))
        x = self.pooling1(x)
        print("pooling1",x.size( ))
        x = F.dropout(x, 0.5)
        print("dropout",x.size( ))
        # Layer 3
        x = self.pointwise(x)
        print("pointwise",x.size( ))
        x = F.elu(self.batchnorm3(x))
        print("batchnorm",x.size( ))
        x = self.pooling2(x)
        print("pooling2",x.size( ))
        x = F.dropout(x, 0.5)
        print("dropout",x.size( ))
        return x
        
       #  FC Layer
        x = F.softmax(self.fc1(x))
        return x
    
from torchsummary import summary
model = EEGNet()
summary(model, input_size=(1, 64, 128))
