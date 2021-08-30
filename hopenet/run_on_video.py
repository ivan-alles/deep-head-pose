""" Run the self-contained Hopenet on a video file. The video shall be cropped to contain the head only. """

import argparse
import glob
import sys
import os
from math import cos, sin

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


def draw_axis_orig(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    """ Orignal Hope code from code/utils.py. Is used for comparison. """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def draw_axes(img, yaw, pitch, roll, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)), size=100, thickness=1):
    """
    Draws axes using rotation matrix.
    :param img: image
    :param yaw: angle prediciton in degrees.
    :param pitch: angle prediciton in degrees.
    :param roll: angle prediciton in degrees.
    :param axes axes unit vectors, default value corresponds to the standard right-handed CS:
    x, y as on the screen, z axis looking away from the observer.
    :param size: axis length.
    :param thickness: line thickness.
    """
    axes = np.array(axes, dtype=np.float32)

    r = hopenet.angles_to_rotation_matrix(yaw, pitch, roll, degrees=True)

    # Make sure this is the rotation matrix (before we create a proper unit test).
    assert np.linalg.norm(np.dot(r, r.T) - np.eye(3)) < 1e-5
    assert np.allclose(np.linalg.det(r), 1)

    origin = np.array((img.shape[1] / 2, img.shape[0] / 2, 0))
    axes = np.dot(axes, r) * size + origin

    o = tuple(origin[:2].astype(int))
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for ai in range(3):
        a = tuple(axes[ai, :2].astype(int))
        cv2.line(img, o, a, colors[ai], thickness)

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

    # Test code
    # yaw = torch.tensor(0.0)
    # pitch = torch.tensor(20.0)
    # roll = torch.tensor(0.0)

    # Show original hopenet version for comparison
    show_original_axes = True

    if show_original_axes:
        # Original assumes that the z axis is looking to the observer (a left-handed CS).
        draw_axis_orig(frame, yaw.item(), pitch.item(), roll.item(), size=100)

    draw_axes(frame, yaw, pitch, roll, size=200, thickness=1)

    cv2.putText(frame, f"-pitch/x {pitch:.2f}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"yaw/y {yaw:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"roll/z: {roll:.2f}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

    if video_out is None:
        video_out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, f'{base_file_name}.avi'), fourcc, 30,
                                    (frame.shape[1::-1]))

    video_out.write(frame)
    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
    frame_num += 1





