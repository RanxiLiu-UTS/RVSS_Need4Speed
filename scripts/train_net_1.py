import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from steerDS import SteerDataSet

torch.manual_seed(0)

# -------------------------
# device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", device)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:, :, ::-1]
    plt.imshow(rgbimg)
    plt.show()

# -------------------------
# dataset / loader
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

script_path = os.path.dirname(os.path.realpath(__file__))

train_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_rx_002'), '.jpg', transform)
val_ds   = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_rx_002'),   '.jpg', transform)

print("The train dataset contains %d images " % len(train_ds))
print("The validation dataset contains %d images " % len(val_ds))

# num_workers/pin_memory（有 CUDA 时加速数据搬运）
pin = (device.type == "cuda")
trainloader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin)
valloader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2, pin_memory=pin)

# -------------------------
# net
# -------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# -------------------------
# training loop
# -------------------------
losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0.0

for epoch in range(10):
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        # move to GPU
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        pred = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_loss = epoch_loss / len(trainloader)
    train_acc = 100.0 * correct / total
    losses['train'].append(train_loss)
    accs['train'].append(train_acc)

    # -------------------------
    # validation
    # -------------------------
    net.eval()
    val_loss = 0.0
    correct_pred = {classname: 0 for classname in val_ds.class_labels}
    total_pred   = {classname: 0 for classname in val_ds.class_labels}

    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            for label, pred in zip(labels, preds):
                name = val_ds.class_labels[label.item()]
                if label == pred:
                    correct_pred[name] += 1
                total_pred[name] += 1

    # class mean accuracy（你的原逻辑）
    class_accs = []
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            continue
        class_accs.append(100.0 * float(correct_count) / total_pred[classname])

    val_acc = float(np.mean(class_accs)) if len(class_accs) else 0.0
    losses['val'].append(val_loss / len(valloader))
    accs['val'].append(val_acc)

    print(f"Epoch {epoch+1:02d} | train loss {train_loss:.4f} acc {train_acc:.2f}% | val loss {losses['val'][-1]:.4f} acc {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), "steer_net.pth")

print("Finished Training. Best val acc:", best_acc)

# -------------------------
# plots
# -------------------------
plt.plot(losses['train'], label='Training')
plt.plot(losses['val'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(accs['train'], label='Training')
plt.plot(accs['val'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -------------------------
# performance evaluation (confusion matrix)
# -------------------------
net.load_state_dict(torch.load("steer_net.pth", map_location=device))
net.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in valloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = net(images)
        preds = outputs.argmax(dim=1)

        actual += labels.cpu().tolist()
        predicted += preds.cpu().tolist()

cm = metrics.confusion_matrix(actual, predicted, normalize='true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_ds.class_labels)
disp.plot()
plt.show()

# per-class accuracy
correct_pred = {classname: 0 for classname in val_ds.class_labels}
total_pred   = {classname: 0 for classname in val_ds.class_labels}
for a, p in zip(actual, predicted):
    cname = val_ds.class_labels[a]
    total_pred[cname] += 1
    if a == p:
        correct_pred[cname] += 1

for classname, correct_count in correct_pred.items():
    if total_pred[classname] == 0:
        continue
    accuracy = 100.0 * correct_count / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')
