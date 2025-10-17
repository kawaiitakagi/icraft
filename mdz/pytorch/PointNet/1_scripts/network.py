"""
@Description: Module to define the network
@Developed by: Alex Choi
@Date: 07/20/2022
@Contact: cinema4dr12@gmail.com
"""

# %% Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    def __init__(self, device='cpu', k=3):
        super().__init__()
        self.device = device
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        # initialize as identity
        self.init = torch.eye(self.k, requires_grad=False).repeat(1, 1, 1).to(self.device)
    def forward(self, _input):
        # input.shape == (bs,n,3)
        bs = _input.size(0)
        xb = F.relu(self.bn1(self.conv1(_input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        # pool = nn.MaxPool1d(xb.size(-1))(xb)
        pool_size = (xb.size(-1),) # for export
        pool = nn.MaxPool1d(pool_size)(xb)# for export
        # pool_size = (1,xb.size(-1)) # for icraft
        # pool = nn.MaxPool2d(pool_size)(xb)# for icraft,将Max_pool1D替换为Max_pool2D
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        # init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1).to(self.device)
        # matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        matrix = self.fc3(xb).view(-1, self.k, self.k) + self.init # for export
        return matrix


class Transform(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.input_transform = Tnet(device, k=3)
        self.feature_transform = Tnet(device, k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, _input):
        matrix3x3 = self.input_transform(_input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(_input, 1, 2), matrix3x3).transpose(1, 2)
        # xb = torch.matmul(torch.transpose(_input, 1, 2), matrix3x3).transpose(1, 2)# for export

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)
        # xb = torch.matmul(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)# for export

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        # xb = nn.MaxPool1d(xb.size(-1))(xb)#
        pool_size = (xb.size(-1),)# for export
        xb = nn.MaxPool1d(pool_size)(xb)# for export
        # pool_size = (1,xb.size(-1))# for export
        # xb = nn.MaxPool2d(pool_size)(xb)# for icraft, 将max_pool1d替换为max_pool2d
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    def __init__(self, num_classes: int, device: str='cpu'):
        super().__init__()
        self.transform = Transform(device)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, _input):
        xb, matrix3x3, matrix64x64 = self.transform(_input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.softmax(output), matrix3x3, matrix64x64 # for export
        # return self.logsoftmax(output), matrix3x3, matrix64x64 
