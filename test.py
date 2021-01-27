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

from torch import optim
from utils.loss import lovasz_hinge


#######################################
# Initializing the arguments for testing
def init_args(args):
    args.train = True
    args.batch_size = 10
    args.developer_mode = True
    args.crf = True

args = get_args()


# change the argumnets for testing
init_args(args)


# ckpt_path = './ckpt'
# exp_name = 'MirrorNet'
# args = {
#     'snapshot': 'MirrorNet',
#     'scale': 384,
#     'crf': True
# }


###############################
# Defining the transoformations
img_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((args.scale, args.scale)),
    transforms.ToTensor()
])
to_test = {'MSD': args.msd_testing_root}

to_pil = transforms.ToPILImage()

###############################
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
    optim_fn = optim.Adam(net.parameters(), lr=args.lr, betas=args.betas)

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

    if args.developer_mode:
        # To include the real images and masks
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform, add_real_imgs=True)
    else:
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform)

    # batch = dataset.sample(3)
    # batch["img"][0].show()
    # batch["mask"][0].show()
    # print(batch)
    
    if args.train:
        print("Training")
        idx = 0
        for i in range(4):
            loss = 0
            batch = dataset.sample(args.batch_size)
            inputs = batch["img"]
            outputs = batch["mask"]
            inputs = torch.from_numpy(inputs)
            outputs = torch.tensor(outputs)

            # To GPU if available
            if args.cuda:
                inputs = inputs.cuda(device_ids[0])
                outputs = outputs.cuda(device_ids[0])

            # Getting the 4 different outputs
            f_4, f_3, f_2, f_1 = net(inputs) 
            f_4 = f_4.data.cpu()
            f_3 = f_3.data.cpu()
            f_2 = f_2.data.cpu()
            f_1 = f_1.data.cpu()
            f_4_arr = []
            f_3_arr = []
            f_2_arr = []
            f_1_arr = []
            f_1_arr_no_resize = []
            for i in range(args.batch_size):
                rev_size = [batch["size"][i][1], batch["size"][i][0]]
                f_4_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_4[i]))))
                f_3_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_3[i]))))
                f_2_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_2[i]))))
                f_1_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_1[i]))))
                f_1_arr_no_resize.append(np.array(to_pil(f_1[i])))
                if args.crf:
                    f_1_arr[i] = crf_refine(np.array(batch["r_img"][i]), f_1_arr[i])
                    # img_ = np.array(batch["img"][i])
                    # img_ =img_.astype('uint8')
                    f_1_arr_no_resize[i] = crf_refine(np.array(batch["r_img"][i].resize((args.scale, args.scale))), f_1_arr_no_resize[i])
                
                image1_size = batch["size"][i]
                if args.save_images:
                    new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                    img_res = Image.fromarray(f_1_arr[i])
                    new_image.paste(batch["r_img"][i],(0,0))
                    new_image.paste(batch["r_mask"][i],(image1_size[0],0))
                    new_image.paste(img_res,(image1_size[0]*2,0))

                    new_image.save(os.path.join(args.msd_results_root, "Testing",
                                                            "MNet_" + str(idx) +".png"))
                    idx +=1

                    img_res1 = Image.fromarray(f_2_arr[i])
                    img_res2 = Image.fromarray(f_3_arr[i])
                    img_res3 = Image.fromarray(f_4_arr[i])
                    new_image.paste(img_res1,(0,0))
                    new_image.paste(img_res2,(image1_size[0],0))
                    new_image.paste(img_res3,(image1_size[0]*2,0))
                    new_image.save(os.path.join(args.msd_results_root, "Testing",
                                                            "MNet_" + str(idx) +".png"))
                idx +=1
            
            # Achieve loss for image with crf refining
            f_1_arr = np.array(f_1_arr)
            f_1_arr_no_resize = np.array(f_1_arr_no_resize)
            f_1_arr_no_resize = torch.tensor(f_1_arr_no_resize)

            loss = lovasz_hinge(torch.tensor(batch["mask"]), f_1_arr_no_resize, per_image=False)
            
        print("Done")
if __name__ == "__main__":
    main()