""" Run the self-contained Hopenet on a video file. The video shall be cropped to contain the head only. """

import argparse
import glob
import sys
import os

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

    if False:
        # Based on 300W-LP with some tweaks (to be explained yet)

        # From 300W-LP RotationMatrix.m
        # function [R] = RotationMatrix(angle_x, angle_y, angle_z)
        # % get rotation matrix by rotate angle
        #
        # phi = angle_x;
        # gamma = angle_y;
        # theta = angle_z;
        #
        # R_x = [1 0 0 ; 0 cos(phi) sin(phi); 0 -sin(phi) cos(phi)];
        # R_y = [cos(gamma) 0 -sin(gamma); 0 1 0; sin(gamma) 0 cos(gamma)];
        # R_z = [cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1];
        #
        # R = R_x * R_y * R_z;
        #
        #
        # end

        phi = np.deg2rad(pitch)   # x
        gamma = -np.deg2rad(yaw)   # y
        theta = -np.deg2rad(roll)  # z
        axes[2] *= -1

        from numpy import sin, cos

        R_x = np.array([[1, 0, 0, ], [0, cos(phi), sin(phi)], [0, -sin(phi), cos(phi)]])
        R_y = np.array([[cos(gamma), 0, -sin(gamma)], [0, 1, 0], [sin(gamma), 0, cos(gamma)]])
        R_z = np.array([[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1]])

        r = np.linalg.multi_dot((R_x, R_y, R_z))
    else:
        # Based on original HopeNet code

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
        cv2.line(img, o, a, colors[ai], 1)




    # size /= 2
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
    transforms.CenterCrop(224), transforms.ToTensor()
])


class VideoReader:
    def __init__(self, video_path):
        self._video = None
        self._frames = None
        self._frames_index = 0
        if os.path.isdir(video_path):
            self._frames = glob.glob(video_path + '/**/*.png', recursive=True)
            self._frames_index = 0
        else:
            self._video = cv2.VideoCapture(video_path)

    def read_frame(self):
        if self._frames is not None:
            if self._frames_index >= len(self._frames):
                return None
            frame_path = self._frames[self._frames_index]
            self._frames_index += 1
            frame = cv2.imread(frame_path)
            return frame
        else:
            ret, frame = self._video.read()
            if not ret:
                frame = None
            return frame


reader = VideoReader(args.video)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

base_file_name = os.path.splitext(os.path.basename(args.video))[0]

txt_out = open(os.path.join(OUTPUT_DIR, f'{base_file_name}.txt'), 'w')
video_out = None

frame_num = 0
while True:
    frame = reader.read_frame()
    if frame is None:
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

    draw_axes(frame, yaw.item(), pitch.item(), roll.item())

    if video_out is None:
        video_out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'{base_file_name}.avi'), fourcc, 30,
                                    (frame.shape[1::-1]))

    video_out.write(frame)
    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
    frame_num += 1





