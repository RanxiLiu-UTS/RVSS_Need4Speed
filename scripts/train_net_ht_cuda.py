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

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################

torch.manual_seed(0)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True


# Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:, :, ::-1]
    plt.imshow(rgbimg)
    plt.show()


#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################

# train_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_01'), '.jpg', transform)
train_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_rx_ht_zh'), '.jpg', transform)

print("The train dataset contains %d images " % len(train_ds))

trainloader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)
# trainloader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

all_y = []
for S in trainloader:
    im, y = S
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

all_lbls, all_counts = np.unique(all_y, return_counts=True)
plt.bar(all_lbls, all_counts, width=(all_lbls[1] - all_lbls[0]) / 2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Training Dataset')
plt.show()

example_ims, example_lbls = next(iter(trainloader))
print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
imshow(torchvision.utils.make_grid(example_ims))


########################
## Validation dataset ##
########################

val_ds = SteerDataSet(os.path.join(script_path, '..', 'data', 'train_01_turning'), '.jpg', transform)
print("The valuation dataset contains %d images " % len(val_ds))

valloader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

all_y = []
for S in valloader:
    im, y = S
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

all_lbls, all_counts = np.unique(all_y, return_counts=True)
plt.bar(all_lbls, all_counts, width=(all_lbls[1] - all_lbls[0]) / 2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Validation Dataset')
plt.show()


#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 128)
        self.fc2 = nn.Linear(128, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)


#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################

losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0

ckpt_path = os.path.join(script_path, '..', 'steer_net.pth')
print(ckpt_path)

for epoch in range(30):

    epoch_loss = 0.0
    correct = 0
    total = 0

    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = epoch_loss / len(trainloader)
    train_acc = 100. * correct / total

    print(f'Epoch {epoch + 1} loss: {train_loss}')

    losses['train'].append(train_loss)
    accs['train'].append(train_acc)

    correct_pred = {classname: 0 for classname in val_ds.class_labels}
    total_pred = {classname: 0 for classname in val_ds.class_labels}

    net.eval()
    val_loss = 0

    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_ds.class_labels[label.item()]] += 1
                total_pred[val_ds.class_labels[label.item()]] += 1

    class_accs = []
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accs.append(accuracy)

    val_acc = np.mean(class_accs)
    losses['val'].append(val_loss / len(valloader))
    accs['val'].append(val_acc)

    if val_acc > best_acc:
        torch.save(net.state_dict(), ckpt_path)
        best_acc = val_acc

print('Finished Training')

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


#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################

net.load_state_dict(torch.load(ckpt_path, map_location=device))
net.to(device)
net.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in val_ds.class_labels}
total_pred = {classname: 0 for classname in val_ds.class_labels}

actual = []
predicted_list = []

with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.detach().cpu().tolist()
        predicted_list += predictions.detach().cpu().tolist()

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[val_ds.class_labels[label.item()]] += 1
            total_pred[val_ds.class_labels[label.item()]] += 1

# cm = metrics.confusion_matrix(actual, predicted_list, normalize='true')
# disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
#                               display_labels=val_ds.class_labels)
# disp.plot()
plt.show()

for classname, correct_count in correct_pred.items():
    if total_pred[classname] > 0:
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')
