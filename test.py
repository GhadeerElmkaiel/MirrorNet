import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

# from config import msd_testing_root, msd_results_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet
from dataset import ImageFolder

from arguments import get_args



args = get_args()

ckpt_path = './ckpt'
exp_name = 'MirrorNet'
# args = {
#     'snapshot': 'MirrorNet',
#     'scale': 384,
#     'crf': True
# }

img_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_test = {'MSD': args.msd_testing_root}

to_pil = transforms.ToPILImage()

# Initializing random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# device_ids = [0]
device_ids = args.device_ids
torch.cuda.set_device(device_ids[0])

# print(args)

def main():
    net = MirrorNet().cuda(device_ids[0])

    if len(args.snapshot) > 0:
        # print(os.path.join(args.root_path + args.ckpt_path, args.exp_name, args.snapshot + '.pth'))
        print('Load snapshot {} for testing'.format(args.snapshot))
        net.load_state_dict(torch.load(os.path.join(args.root_path + args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        # net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        print('Load {} succeed!'.format(os.path.join(args.root_path + args.ckpt_path, args.exp_name, args.snapshot + '.pth')))

    if not args.train:
        net.eval()
        data_path = args.msd_testing_root
    else:
        data_path = args.msd_training_root

    dataset = ImageFolder(data_path, img_transform= img_transform)
    # batch = dataset.sample(3)
    # batch["img"][0].show()
    # batch["mask"][0].show()
    # print(batch)

    if args.train:
        batch = dataset.sample(args.batch_size)
        inputs = batch["img"]
        outputs = batch["mask"]
        inputs = torch.tensor(inputs)
        outputs = torch.tensor(outputs)
        if args.cuda:
            inputs = inputs.cuda(device_ids[0])
            outputs = outputs.cuda(device_ids[0])

if __name__ == "__main__":
    main()