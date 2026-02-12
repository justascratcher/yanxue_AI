import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import pickle

epochs = 16
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.05
momentum = 0.4
log_interval = 40
random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=True,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            )
        ])),
    batch_size=batch_size_train, shuffle=True
)

test_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=False,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            )
        ])),
    batch_size=batch_size_test, shuffle=True
)

example = enumerate(train_loader)
_, (data, target) = next(example)
print(data.shape)
print(target)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 45, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(45, 90, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*3*30, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 64*3*30)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training,p=0.2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training,p=0.2)
        x = F.relu(self.fc4(x))
        return F.log_softmax(x, dim=1)
result=list()
cnt=0
network = CNN()
network.load_state_dict(torch.load('weights/large_16.pth'))
network.to(device)
def train(epoch, lr):
    global cnt
    network.train()
    optimizer = optim.SGD(
        network.parameters(),
        lr=lr,
        momentum=momentum
    )
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        cnt+=target.size(0)
        result.append([cnt, loss.item()]);
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

for epoch in range(epochs):
    train(epoch,learning_rate*math.cos(epoch / epochs * math.pi / 2))
    torch.save(network.state_dict(), f'weights/large_16_(0.05)_{16 if epoch>=8 else 8}.pth')
    with open(f'result_large_{batch_size_train}.pkl', 'wb') as f:
        pickle.dump(result, f)

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