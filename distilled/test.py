""" Test a model on a dataset """

import sys, os, argparse

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from distilled import datasets
from distilled import hopenet
from distilled import utils


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--data_dir', help='Directory path for data.', default='', type=str)
    parser.add_argument('--filename_list',  help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', help='Batch size.', default=1, type=int)
    parser.add_argument('--show-ground-truth', help='Show ground truth axes.', default=False, type=bool)
    parser.add_argument('--show-prediction', help='Show prediction axes.', default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = hopenet.create_model(args.snapshot)
    model.to(device)
    model.eval()

    print('Loading data.')

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), transforms.ToTensor()])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = datasets.AFLW2000_ds(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(args.data_dir, args.filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size=args.batch_size, num_workers=0)


    print('Ready to test network.')

    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    yaw_error = .0
    pitch_error = .0
    roll_error = .0

    rotation_diff_error = 0.

    for i, (images, labels, cont_labels, name) in enumerate(test_loader):
        with torch.no_grad():
            # for image_i in range(len(images)):
            #     image = images[0].moveaxis(0, 2).detach().cpu().numpy()
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #     cv2.imshow(f'image-{i}', image)
            #     cv2.waitKey(0)

            images = Variable(images).to(device)
            total += cont_labels.size(0)

            label_yaw = cont_labels[:, 0].float().to(device)
            label_pitch = cont_labels[:, 1].float().to(device)
            label_roll = cont_labels[:, 2].float().to(device)

            yaw_predicted, pitch_predicted, roll_predicted = model(images)

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

            # Difference between rotation matrices
            rot_pr = utils.hopenet.angles_to_rotation_matrix(yaw_predicted, pitch_predicted, roll_predicted)
            rot_gt = utils.hopenet.angles_to_rotation_matrix(label_yaw, label_pitch, label_roll)
            rotation_diff_error += utils.rotation_diff(rot_gt, rot_pr)

            # Save first image in batch with pose cube or axis.
            if args.show_ground_truth or args.show_prediction:
                name = name[0]
                if args.dataset == 'BIWI':
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '_rgb.png'))
                else:
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                if args.batch_size == 1:
                    error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)),
                                                               torch.sum(torch.abs(pitch_predicted - label_pitch)),
                                                               torch.sum(torch.abs(roll_predicted - label_roll)))
                    cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0] - 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
                if args.show_prediction:
                    utils.draw_axes(cv2_img, yaw_predicted, pitch_predicted, roll_predicted, tx=200, ty=200, size=100)
                if args.show_ground_truth:
                    utils.draw_axes(cv2_img, label_yaw, label_pitch, label_roll, tx=200, ty=200, size=75, thickness=3)

                image_file = os.path.join('output/images', name + '.jpg')
                os.makedirs(os.path.dirname(image_file), exist_ok=True)
                cv2.imwrite(image_file, cv2_img)

    print('Test error in degrees of the model on the ' + str(total) +
    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f Rotation diff: %.4f' % (yaw_error / total,
    pitch_error / total, roll_error / total, rotation_diff_error / total))

    # Profiling code
    print(f'Crop width mean {np.mean(datasets.crop_width)}, crop height mean {np.mean(datasets.crop_height)}')


