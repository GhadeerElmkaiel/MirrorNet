import numpy as np
import os
import time
from tqdm import tqdm

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


# from config import msd_testing_root, msd_results_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet, LitMirrorNet
from dataset import ImageFolder

from arguments import get_args

from utils.loss import lovasz_hinge

#######################################
# Initializing the arguments for testing
def init_args(args):
    args.train = True
    args.batch_size = 20
    args.developer_mode = True
    args.load_model = True
    args.fast_dev_run = False
    args.crf = True
    args.device_ids = [0, 1]
    args.val_every = 5

args = get_args()


#######################################
# Checkpoint call back for saving the best models
# 
checkpoint_callback = ModelCheckpoint(
    monitor= args.monitor,
    dirpath= args.ckpt_path,
    filename= 'MirrorNet-{epoch:02d}-{val_loss:.2f}',
    save_top_k= args.save_top,
    mode='min',
)

tb_logger = pl_loggers.TensorBoardLogger(save_dir = args.log_path,
                                        name = args.log_name)


# change the argumnets for testing
init_args(args)



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


def main():
    net = MirrorNet().cuda(device_ids[0])
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=args.betas)

    if args.load_model:
        # print(os.path.join(args.root_path + args.ckpt_path, args.exp_name, args.snapshot + '.pth'))
        print('Load snapshot {} for testing'.format(args.snapshot))
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        # net.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))
        print('Load {} succeed!'.format(os.path.join(args.ckpt_path, args.exp_name, args.snapshot + '.pth')))

    if not args.train:
        net.eval()
        data_path = args.msd_testing_root
    else:
        data_path = args.msd_training_root
        eval_path = args.msd_eval_root
        net.train()

    if args.developer_mode:
        # To include the real images and masks
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform, add_real_imgs=True)
        eval_dataset = ImageFolder(eval_path, img_transform= img_transform, target_transform=mask_transform, add_real_imgs=True)
    else:
        dataset = ImageFolder(data_path, img_transform= img_transform, target_transform=mask_transform)
        eval_dataset = ImageFolder(eval_path, img_transform= img_transform, target_transform=mask_transform)

    loader = DataLoader(dataset, batch_size= args.batch_size, shuffle=args.shuffle_dataset)
    eval_loader = DataLoader(eval_dataset, batch_size = 1, shuffle=False)

    # batch = dataset.sample(3)
    # batch["img"][0].show()
    # batch["mask"][0].show()
    # print(batch)
    
    if args.train:
        print("Training")

        ##############################
        # Using one GPU                 args.device_ids = [0]
        ##############################
        if len(args.device_ids) == 1:
            idx = 0
            ##############################
            # Training for number of epoches
            for epoch in range(args.epochs):
                start_time = time.time()
                loss = 0

                ###############################
                ## Training
                ###############################
                # Defining the tqdm progress bar for training dataset
                pbar = tqdm(loader, desc ="Processing batch number")
                idx = 0
                net.train()
                for batch in pbar:
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
                    inputs.requires_grad=True
                    outputs.requires_grad=True
                    f_4_gpu, f_3_gpu, f_2_gpu, f_1_gpu = net(inputs)
                    
                    if args.developer_mode:
                        if idx == 0 :
                            f_1 = f_1_gpu.data.cpu()
                            rev_size = [batch["size"][0][1], batch["size"][0][0]]
                            image1_size = batch["size"][0]
                            f_1_trans = np.array(transforms.Resize(rev_size)(to_pil(f_1[0])))
                            f_1_crf = crf_refine(np.array(batch["r_img"][0]), f_1_trans)
                            
                            new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                            img_res = Image.fromarray(f_1_crf)
                            new_image.paste(batch["r_img"][0],(0,0))
                            new_image.paste(batch["r_mask"][0],(image1_size[0],0))
                            new_image.paste(img_res,(image1_size[0]*2,0))

                            new_image.save(os.path.join(args.msd_results_root, "Testing",
                                                                    "Epoch: " + str(epoch) +" Trining.png"))
                            print("Image saved")
                        idx +=1
                    # f_4.requires_grad = True
                    # f_3.requires_grad = True
                    # f_2.requires_grad = True
                    # f_1.requires_grad = True

                    ##############################
                    # # For image processing
                    # f_4 = f_4_gpu.data.cpu()
                    # f_3 = f_3_gpu.data.cpu()
                    # f_2 = f_2_gpu.data.cpu()
                    # f_1 = f_1_gpu.data.cpu()
                    # f_4_arr = []
                    # f_3_arr = []
                    # f_2_arr = []
                    # f_1_arr = []
                    # f_1_arr_no_resize = []
                    # for i in range(args.batch_size):
                    #     rev_size = [batch["size"][i][1], batch["size"][i][0]]
                    #     f_4_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_4[i]))))
                    #     f_3_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_3[i]))))
                    #     f_2_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_2[i]))))
                    #     f_1_arr.append(np.array(transforms.Resize(rev_size)(to_pil(f_1[i]))))
                    #     f_1_arr_no_resize.append(np.array(to_pil(f_1[i])))
                    #     if args.crf and args.developer_mode:
                    #         f_1_arr[i] = crf_refine(np.array(batch["r_img"][i]), f_1_arr[i])
                    #         # img_ = np.array(batch["img"][i])
                    #         # img_ =img_.astype('uint8')
                    #         f_1_arr_no_resize[i] = crf_refine(np.array(batch["r_img"][i].resize((args.scale, args.scale))), f_1_arr_no_resize[i])
                        
                    #     image1_size = batch["size"][i]
                    #     if args.save_images:
                    #         new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                    #         img_res = Image.fromarray(f_1_arr[i])
                    #         new_image.paste(batch["r_img"][i],(0,0))
                    #         new_image.paste(batch["r_mask"][i],(image1_size[0],0))
                    #         new_image.paste(img_res,(image1_size[0]*2,0))

                    #         new_image.save(os.path.join(args.msd_results_root, "Testing",
                    #                                                 "MNet_" + str(idx) +".png"))
                    #         idx +=1

                    #         img_res1 = Image.fromarray(f_2_arr[i])
                    #         img_res2 = Image.fromarray(f_3_arr[i])
                    #         img_res3 = Image.fromarray(f_4_arr[i])
                    #         new_image.paste(img_res1,(0,0))
                    #         new_image.paste(img_res2,(image1_size[0],0))
                    #         new_image.paste(img_res3,(image1_size[0]*2,0))
                    #         new_image.save(os.path.join(args.msd_results_root, "Testing",
                    #                                                 "MNet_" + str(idx) +".png"))
                    #     idx +=1
                    
                    # TODO
                    ## Achieve loss for images with crf refining
                    ## Starting Point
                    # f_1_arr = np.array(f_1_arr)
                    # f_1_arr_no_resize = np.array(f_1_arr_no_resize)
                    # f_1_arr_no_resize = torch.tensor(f_1_arr_no_resize)
                    # loss = lovasz_hinge(torch.tensor(batch["mask"]), f_1_arr_no_resize, per_image=False)

                    loss1 = lovasz_hinge(f_1_gpu, outputs, per_image=False)*args.w_losses[0]
                    loss2 = lovasz_hinge(f_2_gpu, outputs, per_image=False)*args.w_losses[1]
                    loss3 = lovasz_hinge(f_3_gpu, outputs, per_image=False)*args.w_losses[2]
                    loss4 = lovasz_hinge(f_4_gpu, outputs, per_image=False)*args.w_losses[3]
                    loss = loss1 + loss2 + loss3 + loss4
                    # outputs.requires_grad = False
                    # L2 = torch.nn.BCELoss()
                    # loss2 = L2(f_1_gpu, outputs)
                    # loss = loss2 +loss1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    res = loss.data.cpu()

                    text = "epoch: {}, Loss: {}".format(epoch, res)

                    text = text.ljust(40)
                    # text = text + " L_BCE: {}".format(loss2)
                    # text = text.ljust(60)
                    pbar.set_description(text)
                print("Needed time:")
                print(time.time() - start_time )

                ###############################
                ## Evaluation
                ###############################
                # Defining the tqdm progress bar for evaluation dataset
                eval_pbar = tqdm(eval_loader, desc ="Processing batch number")
                idx = 0
                net.eval()
                for batch in eval_pbar:
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
                    inputs.requires_grad=True
                    outputs.requires_grad=True
                    f_4_gpu, f_3_gpu, f_2_gpu, f_1_gpu = net(inputs)
                    
                    if args.developer_mode:
                        if idx == 0 :
                            f_1 = f_1_gpu.data.cpu()
                            rev_size = [batch["size"][0][1], batch["size"][0][0]]
                            image1_size = batch["size"][0]
                            f_1_trans = np.array(transforms.Resize(rev_size)(to_pil(f_1[0])))
                            f_1_crf = crf_refine(np.array(batch["r_img"][0]), f_1_trans)
                            
                            new_image = Image.new('RGB',(3*image1_size[0], image1_size[1]), (250,250,250))
                            img_res = Image.fromarray(f_1_crf)
                            new_image.paste(batch["r_img"][0],(0,0))
                            new_image.paste(batch["r_mask"][0],(image1_size[0],0))
                            new_image.paste(img_res,(image1_size[0]*2,0))

                            new_image.save(os.path.join(args.msd_results_root, "Testing",
                                                                    "Epoch: " + str(epoch) +" Eval.png"))
                            print("Image saved")
                        idx +=1

                    eval_loss = lovasz_hinge(f_1_gpu, outputs, per_image=False)

                    res = eval_loss.data.cpu()

                    text = "epoch: {}, Eval Loss: {}".format(epoch, res)

                    text = text.ljust(45)
                    eval_pbar.set_description(text)

        ##############################
        # Using multiple GPUs                 args.device_ids = [0, 1, ...]
        ##############################
        else:
            # print("1")
            # LitMirrorNet(args).load_from_checkpoint(args.root_path + args.ckpt_path + "/MirrorNet-epoch=16-val_loss=3.99.ckpt")
            # print("2")
            net = LitMirrorNet(args)
            # net = net.load_from_checkpoint(args.root_path + args.ckpt_path + "/MirrorNet-epoch=16-val_loss=3.99.ckpt")
            trainer = Trainer(gpus=args.device_ids,
                            fast_dev_run = args.fast_dev_run,
                            accelerator = 'dp',
                            max_epochs = args.epochs,
                            callbacks = [checkpoint_callback],
                            check_val_every_n_epoch = args.val_every,
                            logger = tb_logger)
                            # resume_from_checkpoint = args.root_path + args.ckpt_path + "/MirrorNet-epoch=16-val_loss=3.99.ckpt")
            trainer.fit(net)
            final_epoch_model_path = args.ckpt_path + "final_epoch.ckpt"
            trainer.save_checkpoint(final_epoch_model_path)

        print("Done")
if __name__ == "__main__":
    main()