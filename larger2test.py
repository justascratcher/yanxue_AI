import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math

batch_size_test = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=False,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            )
        ])),
    batch_size=batch_size_test, shuffle=True
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 45, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(45, 90, kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(90, 180, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*3*60, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv2_drop(self.conv3(x)))
        x = x.view(-1, 64*3*60)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training,p=0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training,p=0.1)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training,p=0.1)
        x = F.relu(self.fc5(x))
        return F.log_softmax(x, dim=1)

network = CNN()
network.load_state_dict(torch.load('weights/large2_16_16_64.pth'))
network.to(device)

def testNN():
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(correct / total)

testNN()