"""
benchmarks/DepthwiseCNN/src/model.py
------------------------------------
PyTorch implementation of the DepthwiseCNN family of architectures:
- DSC (Depthwise Separable Convolution)
- DSC-SE (DSC with Squeeze-and-Excitation)
- M-DSC (Modified DSC)

Reference:
"File Fragment Type Classification Using Light-Weight Convolutional Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Hardswish(nn.Module):
    def forward(self, x):
        return F.hardswish(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, l = x.size()
        y = x.mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def get_norm_layer(norm_type, channels):
    if norm_type == 'batch':
        return nn.BatchNorm1d(channels)
    elif norm_type == 'group':
        # 8 groups works well for 32, 64, 128 channels
        return nn.GroupNorm(8, channels)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

def get_activation_layer(act_type):
    if act_type == 'hardswish':
        return Hardswish()
    elif act_type == 'relu':
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='batch', act_type='hardswish', pool=True):
        super().__init__()
        self.pool = pool
        
        self.branch1 = SeparableConv1d(in_channels, out_channels, kernel_size=11, padding=5)
        self.branch2 = SeparableConv1d(in_channels, out_channels, kernel_size=19, padding=9)
        self.branch3 = SeparableConv1d(in_channels, out_channels, kernel_size=27, padding=13)
        
        self.norm1 = get_norm_layer(norm_type, out_channels)
        self.norm2 = get_norm_layer(norm_type, out_channels)
        self.norm3 = get_norm_layer(norm_type, out_channels)
        
        self.activation = get_activation_layer(act_type)
        
        if self.pool:
            self.max_pool = nn.MaxPool1d(kernel_size=4, stride=4)
            
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        b1 = self.norm1(self.branch1(x))
        b2 = self.norm2(self.branch2(x))
        b3 = self.norm3(self.branch3(x))
        
        out = b1 + b2 + b3
        if self.pool:
            out = self.max_pool(out)
            
        shortcut_out = self.shortcut(x)
        if self.pool:
            shortcut_out = self.max_pool(shortcut_out)
            
        out = out + shortcut_out
        out = self.activation(out)
        return out

class DepthwiseCNNModel(nn.Module):
    def __init__(self, num_classes, variant='dsc'):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        
        if variant not in ['dsc', 'dsc-se', 'm-dsc']:
            raise ValueError(f"Invalid variant '{variant}'. Must be one of 'dsc', 'dsc-se', 'm-dsc'")
            
        norm_type = 'group' if variant == 'm-dsc' else 'batch'
        act_type = 'relu' if variant == 'm-dsc' else 'hardswish'
        use_se = (variant == 'dsc-se')
        
        self.embedding = nn.Embedding(256, 32)
        
        # First layer
        if variant == 'm-dsc':
            self.conv1 = nn.Conv1d(32, 32, kernel_size=19, stride=2, padding=9, groups=32, bias=False)
        else:
            self.conv1 = nn.Conv1d(32, 32, kernel_size=19, stride=2, padding=9, bias=False)
            
        self.norm1 = get_norm_layer(norm_type, 32)
        self.act1 = get_activation_layer(act_type)
        
        # Inception Blocks
        self.block1 = InceptionBlock(32, 64, norm_type, act_type, pool=True)
        self.se1 = SEBlock(64, reduction=2) if use_se else nn.Identity()
        
        self.block2 = InceptionBlock(64, 64, norm_type, act_type, pool=False)
        self.se2 = SEBlock(64, reduction=2) if use_se else nn.Identity()
        
        self.block3 = InceptionBlock(64, 128, norm_type, act_type, pool=True)
        self.se3 = SEBlock(128, reduction=8) if use_se else nn.Identity()
        
        self.dropout = nn.Dropout(0.2) if variant == 'm-dsc' else nn.Identity()
        
        self.classifier = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, length]
        x = self.embedding(x) # [batch, length, 32]
        x = x.transpose(1, 2) # [batch, 32, length]
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.block1(x)
        x = self.se1(x)
        
        x = self.block2(x)
        x = self.se2(x)
        
        x = self.block3(x)
        x = self.se3(x)
        
        x = x.mean(dim=2, keepdim=True) # [batch, 128, 1]
        
        x = self.dropout(x)
        x = self.classifier(x) # [batch, num_classes, 1]
        
        x = x.squeeze(-1) # [batch, num_classes]
        return F.log_softmax(x, dim=1)
