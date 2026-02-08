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
from machinevisiontoolbox import Image


def threshold_stop_sign_image(im_rgb: Image,
                              h1=0.06, h2=0.94,
                              s_min=0.25, v_min=0.25,
                              crop_top=0.25):
    H_img = im_rgb.height
    y0 = int(crop_top * H_img)
    im = im_rgb[y0:, :]

    try:
        im = im.smooth(sigma=1.0)
    except Exception:
        pass

    try:
        hsv = im.colorspace("HSV")
    except Exception:
        try:
            hsv = im.colorconvert("HSV")
        except Exception:
            hsv = im.hsv()

    try:
        h = hsv.plane("H").A.astype(np.float32)
        s = hsv.plane("S").A.astype(np.float32)
        v = hsv.plane("V").A.astype(np.float32)
    except Exception:
        h = hsv.plane(0).A.astype(np.float32)
        s = hsv.plane(1).A.astype(np.float32)
        v = hsv.plane(2).A.astype(np.float32)

    if h.max() > 1.5:
        if h.max() > 180:
            h = h / 360.0
        else:
            h = h / h.max()
    if s.max() > 1.5:
        s = s / 255.0
    if v.max() > 1.5:
        v = v / 255.0

    rgb = im.A.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red_rgb = (R > G + 0.10) & (R > B + 0.10) & (R > 0.18)

    red_h = (h < h1) | (h > h2)
    good_sv = (s > s_min) & (v > v_min)

    mask = red_h & good_sv & red_rgb
    thr = Image(mask.astype(np.uint8) * 255, binary=True)
    return thr, mask.astype(np.uint8)


def stopSignDetection(im_rgb: Image,
                      area_thresh=100, 
                      **thr_kwargs):
    """
    返回: (detected: bool, max_area: float)
    detected=True 表示存在 blob 面积 >= area_thresh
    thr_kwargs 会传给 threshold_stop_sign_image，例如 h1/h2/s_min/v_min/crop_top
    """
    thr, _ = threshold_stop_sign_image(im_rgb, **thr_kwargs)

    try:
        blobs = thr.blobs()
    except Exception:
        return False, 0.0

    # machinevisiontoolbox 的 blob 通常有 .area 属性
    areas = []
    for b in blobs:
        a = getattr(b, "area", None)
        if a is None:
            # 兼容少数版本：用 bbox 估一个面积
            try:
                umin, umax, vmin, vmax = b.bbox  # 如果存在
                a = (umax - umin) * (vmax - vmin)
            except Exception:
                continue
        areas.append(float(a))

    if not areas:
        return False, 0.0

    max_area = max(areas)
    return (max_area >= area_thresh), max_area

def detect_stop_sign(rgb):
    # rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mask = (
        cv2.inRange(hsv, (0,70,70), (10,255,255)) |
        cv2.inRange(hsv, (170,70,70), (180,255,255))
    )

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    binary = Image(mask > 0)

    try:
        blobs = binary.blobs()
    except:
        return False

    return any(b.area > 400 for b in blobs)


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

        self.fc1 = nn.Linear(1344, 128)
        self.fc2 = nn.Linear(128, 5)

        self.relu = nn.ReLU()


    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()

#         # ⭐ 自动推断 fc 输入维度
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, 40, 60)
#             x = self.pool(self.relu(self.conv1(dummy)))
#             x = self.pool(self.relu(self.conv2(x)))
#             feat_dim = x.view(1, -1).shape[1]

#         self.fc1 = nn.Linear(feat_dim, 128)
#         self.fc2 = nn.Linear(128, 5)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

net = Net()
net.eval()

#LOAD NETWORK WEIGHTS HERE
state_dict = torch.load('steer_net.pth', map_location='cpu')
# state_dict = torch.load('steer_net.pth')
net.load_state_dict(state_dict)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])


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


hit_streak = 0
cooldown = 0
COOLDOWN_FRAMES = 50
iteration = 0

stop_until = 0.0       # ✅ 非阻塞停到这个时间点（秒）
STOP_SECONDS = 1.0     # ✅ 停车时长（秒）
iteration = 0

flag = False

cnt = 0

try:
    angle = 0
    while True:
        # get an image from the the robot
        im = bot.getImage()
        # print("im shape: ", im.shape)

        cnt += 1

        if im is None:
            continue

        if flag and cnt < 50:
            continue

        # im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im_bgr = im[120:, :, :]


        #TO DO: apply any necessary image transforms
        im = transform(im_bgr).unsqueeze(0)
        # print("im transform shape: ", im.shape)

        #TO DO: pass image through network get a prediction
        output = net(im)
        _, prediction_label = torch.max(output, 1)
        # print(output)

        #TO DO: convert prediction into a meaningful steering angle
        prediction = 0
        if prediction_label == 0:
            prediction = -0.5
        elif prediction_label == 1:
            prediction = -0.25
        elif prediction_label == 2:
            prediction = 0
        elif prediction_label == 3:
            prediction = 0.25
        elif prediction_label == 4:
            prediction = 0.5

        # if prediction_label == 0:
        #     print(iteration, " sharp left turn")
        # elif prediction_label == 1:
        #     print(iteration, " left turn")
        # elif prediction_label == 2:
        #     print(iteration, " straight")
        # elif prediction_label == 3:
        #     print(iteration, " right turn")
        # elif prediction_label == 4:
        #     print(iteration, " sharp right turn")
        

        iteration += 1

        #TO DO: check for stop signs?
        # thresholded_im = Image(<insert_your_thresholded_im_variable_here>)
        # blobs = thresholded_im.blobs()
        # print(blobs)
        # if detect_stop_sign(im):
        #     bot.setVelocity(0, 0)
        #     continue

        # # (A) 如果正在“停车窗口”内：持续停车，但不阻塞循环
        # now = time.time()
        # if now < stop_until:
        #     bot.setVelocity(0, 0)
        #     # 这里可以选打印，也可以不打印（否则刷屏）
        #     # print(iteration, "STOPPING...")
        #     continue

        # # (B) 停车结束后，进入 cooldown：这段时间不做 stop sign 触发
        # if cooldown > 0:
        #     cooldown -= 1
        # else:
        #     # (C) 正常检测 stop sign
        #     im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        #     im_mvtb = Image(im_rgb)

        #     det, max_area = stopSignDetection(im_mvtb, area_thresh=500)

        #     if det:
        #         hit_streak += 1
        #     else:
        #         hit_streak = 0

        #     # 连续 N 帧检测到才触发
        #     if hit_streak >= 3:
        #         print(f"{iteration} stop sign detected (max_area={max_area:.1f})")

        #         # ✅ 设置“未来 1 秒内都停车”——不阻塞
        #         stop_until = now + STOP_SECONDS

        #         # ✅ 停完后进入 cooldown，避免频繁触发
        #         cooldown = COOLDOWN_FRAMES

        #         hit_streak = 0
        #         bot.setVelocity(0, 0)
        #         continue

        # if cooldown > 0:
        #     cooldown -= 1
        # else:
        #     if iteration % 5 == 0:
        #         im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        #         im_mvtb = Image(im_rgb)
        #         det, max_area = stopSignDetection(im_mvtb, area_thresh=200)
        #         if det:
        #             hit_streak += 1
        #         else:
        #             hit_streak = 0
        #         if hit_streak >= 3:          # 连续3帧都检测到才停
        #             bot.setVelocity(0, 0)
        #             # print("stop sign detected")
        #             time.sleep(1)
        #             cooldown = 60            # 停完后 60 帧内不再触发
        #             hit_streak = 0
        #             continue

        # if cooldown > 0:
        #     cooldown -= 1
        #     # flag = False
        # else:
        #     if iteration % 5 == 0:
        #         im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        #         im_mvtb = Image(im_rgb)
        #         det, max_area = stopSignDetection(im_mvtb, area_thresh=200)
        #         if det:
        #             hit_streak += 1
        #         else:
        #             hit_streak = 0
        #         if hit_streak >= 3:          # 连续3帧都检测到才停
        #             bot.setVelocity(0, 0)
        #             flag = True
        #             # print("stop sign detected")
        #             # time.sleep(1)
        #             cooldown = 60            # 停完后 60 帧内不再触发
        #             hit_streak = 0
        #             continue
        #     else:
        #         flag = False

        
        angle = prediction

        Kd = 20 #base wheel speeds, increase to go faster, decrease to go slower
        Ka = 20 #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
