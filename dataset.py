"""
 @Time    : 10/2/19 18:00
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : dataset.py
 @Function: prepare data for training.
 
"""
import os
import os.path

import torch.utils.data as data
from PIL import Image
import numpy as np


def make_dataset(root):
    img_list = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, 'image')) if f.endswith('.jpg')]
    return [
        (os.path.join(root, 'image', img_name + '.jpg'), os.path.join(root, 'mask', img_name + '.png'))
        for img_name in img_list]


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None, add_real_imgs=False):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.len = len(self.imgs)
        self.add_real_imgs = add_real_imgs

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.len

    def sample(self, batch_size):
        """
        function for getting batch of items of the dataset
        """
        batch = {"img":[], "mask":[], "size":[], "r_img":[], "r_mask":[]}
        indices = np.random.choice(self.len, batch_size, replace=False)
        masks = []
        for i in indices:
            (img, mask) = self.__getitem__(i)
            batch["img"].append(np.asarray(img))
            masks.append(np.asarray(mask))

            img_path, gt_path = self.imgs[i]
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(gt_path)
            
            # Adding the real images to the batch for debugging if needed
            if self.add_real_imgs:
                batch["r_img"].append(img)
                batch["r_mask"].append(mask)
            
            # Adding the real image size to the batch
            w, h = img.size
            batch["size"].append((w, h))
        batch["img"] = np.array(batch["img"])
        # print(masks)
        batch["mask"] = np.asarray(masks)/255.0
        return batch



