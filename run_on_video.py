""" Run the self-contained Hopenet on a video file. The video shall be cropped to contain the head only. """

import sys
import os
import argparse

import cv2
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

import hopenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Test of the self-contained Hopenet implementation')
parser.add_argument('--video', help='Path of video')

args = parser.parse_args()

OUTPUT_DIR = 'output'


def draw_axes(img, yaw, pitch, roll, size=100):
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    y = -np.deg2rad(yaw)
    p = np.deg2rad(pitch)
    r = np.deg2rad(roll)

    # TODO(ia): the matrix (based on the original implementation) is probably changing
    # the axis from right-handed to left-handed.
    r = np.array([
            [np.cos(y) * np.cos(r), -np.cos(y) * np.sin(r), np.sin(y)],
            [np.cos(p) * np.sin(r) + np.cos(r) * np.sin(p) * np.sin(y),
                np.cos(p) * np.cos(r) - np.sin(p) * np.sin(y) * np.sin(r), -np.cos(y) * np.sin(p)],
            [0, 0, 0]  # TODO(ia): find out the last row
    ])

    origin = np.array((img.shape[1] / 2, img.shape[0] / 2, 0))
    axes = np.dot(axes, r.T) * size + origin

    o = tuple(origin[:2].astype(int))
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for ai in range(3):
        a = tuple(axes[ai, :2].astype(int))
        cv2.line(img, o, a, colors[ai], 3)

    # Original implementation

    # pitch = pitch * np.pi / 180
    # yaw = -(yaw * np.pi / 180)
    # roll = roll * np.pi / 180
    #
    # height, width = img.shape[:2]
    # tdx = width / 2
    # tdy = height / 2
    #
    # # X-Axis pointing to right. drawn in red
    # x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    # y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy
    #
    # # Y-Axis | drawn in green
    # #        v
    # x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    # y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy
    #
    # # Z-Axis (out of the screen) drawn in blue
    # x3 = size * (np.sin(yaw)) + tdx
    # y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
    #
    # cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    # cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    # cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(args.video):
    sys.exit('Video does not exist')

model = hopenet.create_model()
model.to(device)
model.eval()

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

video = cv2.VideoCapture(args.video)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

base_file_name = os.path.splitext(os.path.basename(args.video))[0]

out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'{base_file_name}.avi'), fourcc, 30, (width, height))
txt_out = open(os.path.join(OUTPUT_DIR, f'{base_file_name}.txt'), 'w')

frame_num = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform to 224x224
    img = Image.fromarray(rgb_frame)
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).to(device)

    yaw, pitch, roll = model(img)

    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
    draw_axes(frame, yaw.item(), pitch.item(), roll.item())

    out.write(frame)
    frame_num += 1




