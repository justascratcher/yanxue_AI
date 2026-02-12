import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size_test = 1000

testloader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=False, download=True,
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
        self.fc1 = nn.Linear(64*3*60, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
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

model = CNN()
model.load_state_dict(torch.load('weights/large3_16_16_256_(4)_160.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

y_probs = np.concatenate(all_probs, axis=0)
y_true = np.concatenate(all_labels, axis=0)

def visualize_predictions_basic(model, testloader, device, num_images=16):
    model.eval()

    images, labels = next(iter(testloader))
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        confidence = confidences[i].item()

        color = 'green' if preds[i] == labels[i] else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}'
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('CIFAR-10 Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_predictions_basic(model, testloader, device)