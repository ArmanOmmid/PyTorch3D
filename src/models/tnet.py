import torch
import torch.nn as nn
import torch.functional as F

from .modules import LambdaModule

class TNet(nn.Module):
    def __init__(self, k=3, embed_dim=64, layers=3) -> None:
        super().__init__()

        self.k = k

        dim_list = [k] + [embed_dim * (2**i) for i in range(layers)]

        self.conv = []
        for i in range(len(dim_list) - 1):
            self.conv.append(nn.Conv1d(dim_list[i], dim_list[i+1], 1))
            self.conv.append(nn.BatchNorm1d(dim_list[i+1]))
            if i < len(dim_list) - 2:
                self.conv.append(nn.LeakyReLU())
        self.conv = nn.Sequential(*self.conv)
        
        # agnostic to number of points
        self.maxpool = LambdaModule(lambda x: torch.max(x, 2, keepdim=True)[0])

        self.fc = []
        for i in range(len(dim_list) - 2):
            self.fc.append(nn.Linear(dim_list[-i-1], dim_list[-i-2]))
            self.fc.append(nn.BatchNorm1d(dim_list[-i-2]))
            self.fc.append(nn.LeakyReLU())
        self.fc = nn.Sequential(*self.fc)

        self.pose = nn.Linear(embed_dim, k*k)
        self.identity_vector = nn.Parameter(torch.eye(k).flatten(), requires_grad=False)

    def forward(self, x: torch.Tensor):
        
        x = self.conv(x)

        x = self.maxpool(x)
        x = x.squeeze(-1)

        x = self.fc(x)
        x = self.pose(x)
        x += self.identity_vector # rotation is a perturbation from identity
        x = x.view(-1, self.k, self.k)
        return x
