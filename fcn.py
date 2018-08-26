import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/pochih/FCN-pytorch
# https://github.com/wkentaro/pytorch-fcn


class FCN_64_11(nn.Module):
    def __init__(self, in_channels,):
        super(FCN_64_11, self).__init__()
    
        # relu = nn.ReLU(inplace=True)
        relu = nn.LeakyReLU(0.1, inplace=True)
        
        # (64, 64) => (54, 54)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11),
            nn.BatchNorm2d(96),
            relu,
        )

        # (54, 54) => (26, 26)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            relu,
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            relu,
            nn.Conv2d(96, 96, kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            relu,
        )

        # (26, 26) => (11, 11)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            relu,
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            relu,
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            relu,
        )

        #　(11, 11) => (11, 11)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            relu,
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
        )

        self.out_conv = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        # expected input shape: (64, 64)
        # expected output shape: (11, 11)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_conv(x)
        return x
    

class FCN_112_11(nn.Module):
    def __init__(self, in_channels,):
        super(FCN_112_11, self).__init__()
    
        # relu = nn.ReLU(inplace=True)
        relu = nn.LeakyReLU(0.2, inplace=True)
        
        # (112, 112) => (56, 56)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            relu,
        )

        # (56, 56) => (26, 26)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            relu,
        )

        # (26, 26) => (11, 11)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            relu,
        )
        
        #　(11, 11) => (11, 11)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
        )

        self.out_conv = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        # expected input shape: (64, 64)
        # expected output shape: (11, 11)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out_conv(x)
        return x
    

class FCN_224_11(nn.Module):
    def __init__(self, in_channels,):
        super(FCN_224_11, self).__init__()
    
        # relu = nn.ReLU(inplace=True)
        relu = nn.LeakyReLU(0.1, inplace=True)
        
        # (224, 224) => (112, 112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            relu,
        )

        # (112, 112) => (56, 56)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            relu,
        )

        # (56, 56) => (26, 26)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            relu,
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            relu,
        )

        # (26, 26) => (11, 11)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            relu,
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            relu,
        )
        
        #　(11, 11) => (11, 11)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            relu,
        )

        self.out_conv = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        # expected input shape: (64, 64)
        # expected output shape: (11, 11)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out_conv(x)
        return x