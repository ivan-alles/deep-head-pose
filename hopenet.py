""" A self-contained one-file version of the HopeNet. """

import os
import math

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import logging
from urllib import request

logger = logging.getLogger(__name__)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def create_model():
    model_dir = 'models'
    model_file = 'hopenet_robust_alpha1.pkl'
    # The model recommended by the author for practical usage.
    model_path = os.path.join(model_dir, model_file)

    if not os.path.isfile(model_path):
        logger.info('Downloading Hopenet model...')
        request.urlretrieve('https://drive.google.com/u/0/uc?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR&export=download', model_path)

    saved_state_dict = torch.load(model_path)
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # These was added to the model after the checkpoint had been saved.
    saved_state_dict['idx_tensor'] = torch.tensor(range(66))
    saved_state_dict['mean'] = torch.Tensor(MEAN).reshape((1, 3, 1, 1))
    saved_state_dict['std'] = torch.Tensor(STD).reshape((1, 3, 1, 1))

    model.load_state_dict(saved_state_dict)

    return model


class Hopenet(nn.Module):
    """
        Hopenet with 3 output layers for yaw, pitch and roll.
        Predicts Euler angles by binning and regression with the expected value.
        The input shall be an N, 3, 224, 224 tensor in RGB format.
    """
    def __init__(self, block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_bins=66):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.idx_tensor = nn.Parameter(torch.tensor(range(66)), requires_grad=False)
        self.mean = nn.Parameter(data=torch.Tensor(MEAN).reshape((1, 3, 1, 1)), requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(STD).reshape((1, 3, 1, 1)), requires_grad=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        yaw = F.softmax(yaw, dim=1)
        pitch = F.softmax(pitch, dim=1)
        roll = F.softmax(roll, dim=1)

        # Get continuous predictions in degrees.
        yaw = torch.sum(yaw * self.idx_tensor) * 3 - 99
        pitch = torch.sum(pitch * self.idx_tensor) * 3 - 99
        roll = torch.sum(roll * self.idx_tensor) * 3 - 99

        return yaw, pitch, roll