import argparse
import math
import torch
import os 


def get_args():
    root_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='MirrorNet')

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--snapshot', type=str, default="MirrorNet",
                        help='Name of the snapshot to load (default: MirrorNet)')
    parser.add_argument('--root_path', type=str, default=root_path,
                        help='the root path (default: {})'.format(root_path))
    parser.add_argument('--ckpt_path', type=str, default="/ckpt",
                        help='Path to the checkpoints (default: /ckpt)')
    parser.add_argument('--exp_name', type=str, default="MirrorNet",
                        help='Name of the folder of the snapshot to load (default: MirrorNet)')
    parser.add_argument('--scale', type=int, default=384,
                        help='Image scale  (default: 384)')
    parser.add_argument('--crf', action='store_true', default=False,
                        help='use crf (default: False)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train the model (default: False)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size (default: 10)')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--backbone_path', type=str, default=root_path+"/backbone/resnext/resnext_101_32x4d.pth",
                        help='Path to the backbone (default:'+ root_path+'/backbone/resnext/resnext_101_32x4d.pth')
    parser.add_argument('--msd_training_root', type=str, default=root_path+"/MSD/train",
                        help='Path to the training data (default: '+root_path+'/MSD/train')
    parser.add_argument('--msd_testing_root', type=str, default=root_path+"/MSD/test",
                        help='Path to the testing data (default: '+root_path+'/MSD/test')
    parser.add_argument('--msd_results_root', type=str, default=root_path+"/MSD/results",
                        help='Path results path (default: '+root_path+'/MSD/results')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args



