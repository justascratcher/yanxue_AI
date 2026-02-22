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
import torch.optim.lr_scheduler as lr_scheduler
importfile = "160"
epochs = 160
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.1
momentum = 0.4
log_interval = 120
random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=True,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=5),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.232, 0.261)
            )
        ])),
    batch_size=batch_size_train, shuffle=True
)

test_loader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=False,download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.232, 0.261)
            )
        ])),
    batch_size=batch_size_test, shuffle=True
)

example = enumerate(train_loader)
_, (data, target) = next(example)
print(data.shape)
print(target)

img = data[0].cpu().numpy().transpose((1, 2, 0))
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
img = std * img + mean
img = np.clip(img, 0, 1)
plt.imshow(img)
plt.show()

class SENet(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(100, 200, kernel_size=5,padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(64*200, 10)
        self.se = SENet(in_channel=100, reduction=16)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.se(x)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 64*200)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
result=list()
cnt=0
network = CNN()
if importfile != "":
    network.load_state_dict(torch.load(f'weights/small_{importfile}.pth'))
network.to(device)

optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,
    momentum=momentum
)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=64000)
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
        result.append([cnt, loss.item()])
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
    torch.save(network.state_dict(), f'weights/small_{importfile}{'' if importfile=='' else '_'}{160 if epoch>=80 else 80}.pth')
    with open(f'result_small_{importfile}{'' if importfile=='' else '_'}{batch_size_train}.pkl', 'wb') as f:
        pickle.dump(result, f)
    if (epoch+1) % 10 == 0 or math.pow(2,math.ceil(math.log2(epoch+1)))==epoch+1:
        testNN()

testNN()