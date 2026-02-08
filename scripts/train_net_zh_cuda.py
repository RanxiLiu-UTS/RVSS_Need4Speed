import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import matplotlib.pyplot as plt

from steerDS import SteerDataSet

torch.manual_seed(0)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True

# ================= DATA TRANSFORM =================
# ⚠️ 重点：强数据增强，强行打破“跑道记忆”
train_transform = transforms.Compose([
    transforms.Resize((40, 60)),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.05
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
    transforms.Resize((40, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

script_path = os.path.dirname(os.path.realpath(__file__))

# ================= DATASET =================
train_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'train_01_hotspot_40000imgs'),
    '.jpg',
    train_transform
)
print("Train samples:", len(train_ds))

trainloader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

val_ds = SteerDataSet(
    os.path.join(script_path, '..', 'data', 'val_starter'),
    '.jpg',
    val_transform
)
print("Val samples:", len(val_ds))

valloader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

# ================= MODEL =================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # ⭐ 自动推断 fc 输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 40, 60)
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            feat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)

# ================= LOSS & OPTIM =================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ⭐ 模糊类别边界
optimizer = optim.Adam(
    net.parameters(),
    lr=1e-3,
    weight_decay=1e-4       # ⭐ 防止记跑道
)

# ================= TRAINING =================
num_epochs = 30
best_acc = 0.0
ckpt_path = os.path.join(script_path, '..', 'steer_net.pth')

losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total

    losses['train'].append(train_loss)
    accs['train'].append(train_acc)

    # ================= VALIDATION =================
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(valloader)
    val_acc = 100. * correct / total

    losses['val'].append(val_loss)
    accs['val'].append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), ckpt_path)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | '
          f'Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%')

print("Training finished. Best Val Acc:", best_acc)

# ================= CURVE =================
plt.figure()
plt.plot(losses['train'], label='Train')
plt.plot(losses['val'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(accs['train'], label='Train')
plt.plot(accs['val'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()