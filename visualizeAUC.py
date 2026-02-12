import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

batch_size_test = 1000

testloader = DataLoader(torchvision.datasets.CIFAR10('./data/', train=False,download=True,
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


def plot_cifar10_roc_split(model, testloader, device):
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


    n_classes = 10
    y_true_bin = label_binarize(y_true, classes=range(n_classes))


    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('One-vs-Rest ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)



    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true_bin[:, i], y_probs[:, i])[0]
        for i in range(n_classes)
    ]))


    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)


    class_aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        class_aucs.append(auc(fpr, tpr))


    ax2.plot(all_fpr, mean_tpr, 'b-', lw=3,
             label=f'Macro-average ROC (AUC = {macro_auc:.4f})')

    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Macro-average ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)


    stats_text = "Class AUCs:\n" + "\n".join([
        f"{class_names[i]}: {class_aucs[i]:.3f}"
        for i in np.argsort(class_aucs)[::-1][:5]
    ])
    ax2.text(0.98, 0.02, stats_text,
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('ROC', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_cifar10_roc_split(model, testloader, device)