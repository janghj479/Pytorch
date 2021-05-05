import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
                
        # Layer 1
        self.conv2D = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), padding = "same")
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        
        # Layer 2
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(8,1), 
                                                                          padding ="valid", groups=8)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling1 = nn.AvgPool2d(1, 4)
        
        # Layer 3
        self.pointwise = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d(1, 8)
        
        # FC Layer
        self.fc1 = nn.Linear(4*2*7, 1)
        

    def forward(self, x):
        
        # Layer 2
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = F.elu(self.batchnorm2(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)
        
        # Layer 3
        x = self.pointwise(x)
        x = F.elu(self.batchnorm3(x))
        x = self.pooling2(x)
        x = F.dropout(x, 0.5)
        
        # FC Layer
        x = x.view(-1, 4*2*7)
        x = F.softmax(self.fc1(x))
        return x