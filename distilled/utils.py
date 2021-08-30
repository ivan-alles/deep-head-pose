import cv2
import numpy as np
import torch

from distilled import hopenet


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def draw_axis_orig(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    """ Orignal Hope code from code/utils.py. Is used for comparison. """
    from math import sin, cos

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
