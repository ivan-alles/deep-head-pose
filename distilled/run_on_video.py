""" Run the self-contained Hopenet on a video file. The video shall be cropped to contain the head only. """

import argparse
import glob
import sys
import os

import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from distilled import hopenet
from distilled import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Test of the self-contained Hopenet implementation')
parser.add_argument('--video', help='Path of video')

args = parser.parse_args()

OUTPUT_DIR = 'output'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            if video_path.isnumeric():
                video_path = int(video_path)  # Camera id
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

    # Test code
    # yaw = torch.tensor(0.0)
    # pitch = torch.tensor(20.0)
    # roll = torch.tensor(0.0)

    # Show original hopenet version for comparison
    show_original_axes = True

    if show_original_axes:
        # Original assumes that the z axis is looking to the observer (a left-handed CS).
        utils.draw_axis_orig(frame, yaw.item(), pitch.item(), roll.item(), size=100)

    utils.draw_axes(frame, yaw, pitch, roll, size=200, thickness=1)

    cv2.putText(frame, f"-pitch/x {pitch:.2f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"yaw/y {yaw:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"roll/z: {roll:.2f}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)

    if video_out is None:
        video_out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'{base_file_name}.avi'), fourcc, 30,
                                    (frame.shape[1::-1]))

    video_out.write(frame)
    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
    frame_num += 1





