import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(1*4*16,1)
        

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
summary(model, input_size=(None, 64, 128))

#%% Summary
"""
input torch.Size([2, 1, 64, 128])
conv2D torch.Size([2, 8, 64, 65])
batchnorm torch.Size([2, 8, 64, 65])
depthwise torch.Size([2, 16, 57, 65])
batchnorm torch.Size([2, 16, 57, 65])
pooling1 torch.Size([2, 16, 15, 17])
dropout torch.Size([2, 16, 15, 17])
pointwise torch.Size([2, 16, 15, 2])
batchnorm torch.Size([2, 16, 15, 2])
pooling2 torch.Size([2, 16, 2, 1])
dropout torch.Size([2, 16, 2, 1])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 64, 65]             520
       BatchNorm2d-2            [-1, 8, 64, 65]              16
            Conv2d-3           [-1, 16, 57, 65]             144
       BatchNorm2d-4           [-1, 16, 57, 65]              32
         AvgPool2d-5           [-1, 16, 15, 17]               0
            Conv2d-6            [-1, 16, 15, 2]           4,112
       BatchNorm2d-7            [-1, 16, 15, 2]              32
         AvgPool2d-8             [-1, 16, 2, 1]               0
================================================================
Total params: 4,856
Trainable params: 4,856
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.03
Forward/backward pass size (MB): 1.45
Params size (MB): 0.02
Estimated Total Size (MB): 1.50
----------------------------------------------------------------
"""
