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

epochs = 164
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.4
log_interval = 40
random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=True,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64*128, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x), 2)
        x = F.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        x = x.view(-1, 64*128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
result=list()
cnt=0
network = CNN()
network.load_state_dict(torch.load('weights/weights_(2)_82.pth'))
network.to(device)
optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=2e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64000)
def train(epoch):
    global cnt
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        cnt+=target.size(0)
        result.append([cnt, loss.item()]);
        if batch_idx % log_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

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

for epoch in range(epochs):
    train(epoch)
    torch.save(network.state_dict(), f'weights/weights_(2)_{164 if epoch>=82 else 82}.pth')
    with open(f'result_(2)_{batch_size_train}.pkl', 'wb') as f:
        pickle.dump(result, f)
    if int(math.pow(2,math.log2(epoch+1))) == epoch+1:
        testNN()



testNN()