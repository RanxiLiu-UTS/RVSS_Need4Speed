#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot



parser = argparse.ArgumentParser(description='PiBot client')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
args = parser.parse_args()

bot = PiBot(ip=args.ip)

# stop the robot 
bot.setVelocity(0, 0)

#INITIALISE NETWORK HERE
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

#LOAD NETWORK WEIGHTS HERE
# weights_path = os.path.join(script_path, "steer_net.pth")  # 1) 把权重放在脚本同目录最省事
# 如果 steer_net.pth 在别的地方，用绝对路径/相对路径改这里：
# weights_path = os.path.abspath(os.path.join(script_path, "..", "steer_net.pth"))
repo_root = os.path.abspath(os.path.join(script_path, ".."))
weights_path = os.path.join(repo_root, "steer_net.pth")

state = torch.load(weights_path, map_location=device)
net.load_state_dict(state)
net.eval()
print(f"[OK] Loaded weights: {weights_path} on {device}")

# --- image preprocessing (must match training) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((40, 60)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# class -> steering angle (match 5 classes)
# 0: sharp left, 1: left, 2: straight, 3: right, 4: sharp right
cls_to_angle = {0: -0.5, 1: -0.25, 2: 0.0, 3: 0.25, 4: 0.5}

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

try:
    angle = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()

        #TO DO: apply any necessary image transforms
        

        #TO DO: pass image through network get a prediction

        #TO DO: convert prediction into a meaningful steering angle

        #TO DO: check for stop signs?
        
        # 1) apply same preprocessing as training
        im_crop = im[120:, :, :]                        # crop top 120 rows
        x = transform(im_crop).unsqueeze(0).to(device)  # [1,3,40,60]

        # 2) forward pass -> predicted class
        with torch.no_grad():
            logits = net(x)                             # [1,5]
            pred_cls = int(torch.argmax(logits, dim=1).item())

        # 3) class -> steering angle
        angle = cls_to_angle[pred_cls]

        thresholded_im = Image(<insert_your_thresholded_im_variable_here>)
        blobs = thresholded_im.blobs()
        print(blobs)

        # angle = 0

        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 20 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
