import torch
import torch.nn as nn
import torch.functional as F
from torchvision.ops.misc import Permute

from .tnet import TNet
from .modules import LambdaModule

class PointNet(nn.Module):
    def __init__(self, classes, embed_dim=64, layers=3) -> None:
        super().__init__()
        
        self.permute = Permute([0, 2, 1])

        self.tnet1 = TNet(3, 64)

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.tnet2 = TNet(64, 64)

        dims_list = [embed_dim * (2**i) for i in range(layers)]

        self.conv2 = []
        for i in range(len(dims_list) - 1):
            self.conv2.append(nn.Conv1d(dims_list[i], dims_list[i+1], 1))
            self.conv2.append(nn.BatchNorm1d(dims_list[i+1]))
            if i < len(dims_list) - 2:
                self.conv2.append(nn.LeakyReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.maxpool = LambdaModule(lambda x: torch.max(x, 2, keepdim=True)[0])

        self.fc = []
        for i in range(len(dims_list) - 2):
            if dims_list[-i-2] <= classes:
                break # Don't downsample below number of classes
            self.fc.append(nn.Linear(dims_list[-i-1], dims_list[-i-2]))
            self.fc.append(nn.BatchNorm1d(dims_list[-i-2]))
            self.fc.append(nn.LeakyReLU())
        self.fc = nn.Sequential(*self.fc)

        self.head = nn.Linear(dims_list[-i-2], classes)

    def forward(self, x: torch.Tensor):

        # B L C or B L 3

        x = self.permute(x) # Conv dims B C L

        T1 = self.tnet1(x)
        x = self.permute(torch.bmm(self.permute(x), T1))

        x = self.conv1(x)

        T2 = self.tnet2(x)
        x = self.permute(torch.bmm(self.permute(x), T2))

        x = self.conv2(x)

        x = self.maxpool(x) # global info : B, C, 1

        x = x.squeeze(-1)

        x = self.fc(x)

        x = self.head(x)

        return x


        
