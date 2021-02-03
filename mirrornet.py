import os
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from misc import check_mkdir, crf_refine
from dataset import ImageFolder
from backbone.resnext.resnext101_regular import ResNeXt101

import pytorch_lightning as pl
from utils.optimizer import get_optim
from PIL import Image

from utils.loss import lovasz_hinge

to_pil = transforms.ToPILImage()


# ###################################################################
# # ######################### Upsamle ###############################
# ###################################################################
# class Upsample(nn.Module):
#     def __init__(self,  scale_factor):
#         super(Upsample, self).__init__()
#         self.scale_factor = scale_factor
#     def forward(self, x):
#         return F.interpolate(x, scale_factor=self.scale_factor)


###################################################################
# ########################## CBAM #################################
###################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


###################################################################
# ###################### Contrast Module ##########################
###################################################################
class Contrast_Module(nn.Module):
    def __init__(self, planes):
        super(Contrast_Module, self).__init__()
        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.outplanes = int(planes / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(self.inplanes_half, self.outplanes, 3, 1, 1),
                                  nn.BatchNorm2d(self.outplanes), nn.ReLU())

        self.contrast_block_1 = Contrast_Block(self.outplanes)
        self.contrast_block_2 = Contrast_Block(self.outplanes)
        self.contrast_block_3 = Contrast_Block(self.outplanes)
        self.contrast_block_4 = Contrast_Block(self.outplanes)

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        contrast_block_1 = self.contrast_block_1(conv2)
        contrast_block_2 = self.contrast_block_2(contrast_block_1)
        contrast_block_3 = self.contrast_block_3(contrast_block_2)
        contrast_block_4 = self.contrast_block_4(contrast_block_3)

        output = self.cbam(torch.cat((contrast_block_1, contrast_block_2, contrast_block_3, contrast_block_4), 1))

        return output


###################################################################
# ###################### Contrast Block ###########################
###################################################################
class Contrast_Block(nn.Module):
    def __init__(self, planes):
        super(Contrast_Block, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 4)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4)

        self.local_3 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_3 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8)

        self.local_4 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_4 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=16, dilation=16)

        self.bn = nn.BatchNorm2d(self.outplanes)
        self.relu = nn.ReLU()

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2(x)
        context_2 = self.context_2(x)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn(ccl_2)
        ccl_2 = self.relu(ccl_2)

        local_3 = self.local_3(x)
        context_3 = self.context_3(x)
        ccl_3 = local_3 - context_3
        ccl_3 = self.bn(ccl_3)
        ccl_3 = self.relu(ccl_3)

        local_4 = self.local_4(x)
        context_4 = self.context_4(x)
        ccl_4 = local_4 - context_4
        ccl_4 = self.bn(ccl_4)
        ccl_4 = self.relu(ccl_4)

        output = self.cbam(torch.cat((ccl_1, ccl_2, ccl_3, ccl_4), 1))

        return output


###################################################################
# ########################## NETWORK ##############################
###################################################################
class MirrorNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(MirrorNet, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.contrast_4 = Contrast_Module(2048)
        self.contrast_3 = Contrast_Module(1024)
        self.contrast_2 = Contrast_Module(512)
        self.contrast_1 = Contrast_Module(256)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Conv2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(256)
        self.cbam_2 = CBAM(128)
        self.cbam_1 = CBAM(64)

        self.layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        contrast_4 = self.contrast_4(layer4)
        up_4 = self.up_4(contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)
        layer4_map = torch.sigmoid(layer4_predict)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        up_3 = self.up_3(contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)
        layer3_map = torch.sigmoid(layer3_predict)

        contrast_2 = self.contrast_2(layer2 * layer3_map)
        up_2 = self.up_2(contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)
        layer2_map = torch.sigmoid(layer2_predict)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        up_1 = self.up_1(contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            m = torch.nn.ReLU()
            # return layer4_predict, layer3_predict, layer2_predict, layer1_predict
            return m(layer4_predict), m(layer3_predict), m(layer2_predict), m(layer1_predict)

        return torch.sigmoid(layer4_predict), torch.sigmoid(layer3_predict), torch.sigmoid(layer2_predict), \
               torch.sigmoid(layer1_predict)


###################################################################
# ###################### LIGHTNINH NETWORK ########################
###################################################################

test_iter = 0

class LitMirrorNet(pl.LightningModule):
    def __init__(self, args, backbone_path=None):
        super(LitMirrorNet, self).__init__()
        # global test_iter
        # test_iter = 0
        resnext = ResNeXt101(backbone_path)
        self.val_iter = 0
        self.test_iter = 0
        self.args = args
        self.testing_path = args.msd_testing_root
        self.training_path = args.msd_training_root
        self.eval_path = args.msd_eval_root

        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.contrast_4 = Contrast_Module(2048)
        self.contrast_3 = Contrast_Module(1024)
        self.contrast_2 = Contrast_Module(512)
        self.contrast_1 = Contrast_Module(256)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Conv2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(256)
        self.cbam_2 = CBAM(128)
        self.cbam_1 = CBAM(64)

        self.layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        ###############################
        # Defining the transoformations
        self.img_transform = transforms.Compose([
            transforms.Resize((self.args.scale, self.args.scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.args.scale, self.args.scale)),
            transforms.ToTensor()
        ])


    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        contrast_4 = self.contrast_4(layer4)
        up_4 = self.up_4(contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)
        layer4_map = torch.sigmoid(layer4_predict)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        up_3 = self.up_3(contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)
        layer3_map = torch.sigmoid(layer3_predict)

        contrast_2 = self.contrast_2(layer2 * layer3_map)
        up_2 = self.up_2(contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)
        layer2_map = torch.sigmoid(layer2_predict)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        up_1 = self.up_1(contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        layer4_predict = F.interpolate(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.interpolate(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.interpolate(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.interpolate(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict

        return torch.sigmoid(layer4_predict), torch.sigmoid(layer3_predict), torch.sigmoid(layer2_predict), \
               torch.sigmoid(layer1_predict)

    ###############################################
    # Ligtning functions
    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = batch[1]
        # inputs = torch.from_numpy(inputs)
        # outputs = torch.tensor(outputs)

        inputs.requires_grad=True
        outputs.requires_grad=True
        f_4_gpu, f_3_gpu, f_2_gpu, f_1_gpu = self(inputs)

        loss1 = lovasz_hinge(f_1_gpu, outputs, per_image=False)*self.args.w_losses[0]
        loss2 = lovasz_hinge(f_2_gpu, outputs, per_image=False)*self.args.w_losses[1]
        loss3 = lovasz_hinge(f_3_gpu, outputs, per_image=False)*self.args.w_losses[2]
        loss4 = lovasz_hinge(f_4_gpu, outputs, per_image=False)*self.args.w_losses[3]
        loss = loss1 + loss2 + loss3 + loss4
        self.log('train_loss', loss)
        # self.logger.experiment.log('train_loss', loss)
        # # This does not work
        # # This give back the error: RuntimeError: grad can be implicitly created only for scalar outputs
        # return {'loss': loss}
        return loss

    def configure_optimizers(self):
        optimizer = get_optim(self, self.args)
        return optimizer

    def train_dataloader(self):
        # if self.args.developer_mode:
        #     # To include the real images and masks
        #     dataset = ImageFolder(self.training_path, img_transform= self.img_transform, target_transform= self.mask_transform, add_real_imgs=True)
        # else:
        #     dataset = ImageFolder(self.training_path, img_transform= self.img_transform, target_transform= self.mask_transform)
        dataset = ImageFolder(self.training_path, img_transform= self.img_transform, target_transform= self.mask_transform)
        
        loader = DataLoader(dataset, batch_size= self.args.batch_size, num_workers = 4, shuffle=self.args.shuffle_dataset)

        return loader

    def val_dataloader(self):
        # if self.args.developer_mode:
        #     # To include the real images and masks
        #     eval_dataset = ImageFolder(self.eval_path, img_transform= self.img_transform, target_transform= self.mask_transform, add_real_imgs=True)
        # else:
        #     eval_dataset = ImageFolder(self.eval_path, img_transform= self.img_transform, target_transform= self.mask_transform)
        
        eval_dataset = ImageFolder(self.eval_path, img_transform= self.img_transform, target_transform= self.mask_transform)
        loader = DataLoader(eval_dataset, batch_size= self.args.eval_batch_size, num_workers = 4, shuffle=False)
        self.eval_set = eval_dataset
        return loader

    def test_dataloader(self):
        test_dataset = ImageFolder(self.testing_path, img_transform= self.img_transform, target_transform= self.mask_transform, add_real_imgs = (self.args.developer_mode and not self.args.train))
        loader = DataLoader(test_dataset, batch_size= self.args.test_batch_size, num_workers = 4, shuffle=False)

        return loader

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        outputs = batch[1]
        # inputs = torch.from_numpy(inputs)
        # outputs = torch.tensor(outputs)

        inputs.requires_grad=True
        outputs.requires_grad=True
        f_4_gpu, f_3_gpu, f_2_gpu, f_1_gpu = self(inputs)

        loss1 = lovasz_hinge(f_1_gpu, outputs, per_image=False)*self.args.w_losses[0]
        loss2 = lovasz_hinge(f_2_gpu, outputs, per_image=False)*self.args.w_losses[1]
        loss3 = lovasz_hinge(f_3_gpu, outputs, per_image=False)*self.args.w_losses[2]
        loss4 = lovasz_hinge(f_4_gpu, outputs, per_image=False)*self.args.w_losses[3]
        loss = loss1 + loss2 + loss3 + loss4
        self.log('val_loss', loss)
        # self.logger.experiment.log('val_loss', loss)
        return {'val_loss': loss}
        # return loss


    def test_step(self, batch, batch_idx):
        global test_iter
        inputs = batch[0]
        outputs = batch[1]
        # inputs = torch.from_numpy(inputs)
        # outputs = torch.tensor(outputs)
        f_4_gpu, f_3_gpu, f_2_gpu, f_1_gpu = self(inputs)

        loss1 = lovasz_hinge(f_1_gpu, outputs, per_image=False)*self.args.w_losses[0]
        loss2 = lovasz_hinge(f_2_gpu, outputs, per_image=False)*self.args.w_losses[1]
        loss3 = lovasz_hinge(f_3_gpu, outputs, per_image=False)*self.args.w_losses[2]
        loss4 = lovasz_hinge(f_4_gpu, outputs, per_image=False)*self.args.w_losses[3]
        loss = loss1 + loss2 + loss3 + loss4
        self.log('test_loss', loss)
        if self.args.developer_mode:
            real_img = batch[2]
            real_mask = batch[3]
            sq_zero  = real_img[0].squeeze()
            sq_zero = sq_zero.cpu().numpy()
            real_img = Image.fromarray(sq_zero)

            sq_m  = real_mask[0].squeeze()
            sq_m = sq_m.cpu().numpy()
            real_mask = Image.fromarray(sq_m)

            im_size = real_img.size
            rev_size = [im_size[1], im_size[0]]
            f_1 = f_1_gpu.data.cpu()
            f_1_trans = np.array(transforms.Resize(rev_size)(to_pil(f_1[0])))
            f_1_crf = crf_refine(np.array(real_img), f_1_trans)
            new_image = Image.new('RGB',(3*im_size[0], im_size[1]), (250,250,250))
            img_res = Image.fromarray(f_1_crf)
            new_image.paste(real_img,(0,0))
            new_image.paste(real_mask,(im_size[0],0))
            new_image.paste(img_res,(im_size[0]*2,0))

            # The number of test itteration
            # self.test_iter +=1 
            test_iter +=1 
            new_image.save(os.path.join(self.args.msd_results_root, "Testing",
                                                    "image: " + str(test_iter) +" test.png"))

        # self.logger.experiment.log('val_loss', loss)
        return {'test_loss': loss}
        # return loss
    
    # Function which is activated when the validation epoch wnds
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.args.developer_mode:
            batch = self.eval_set.sample(1)
            inputs = batch["img"]
            outputs = batch["mask"]
            inputs = torch.from_numpy(inputs)
            outputs = torch.tensor(outputs)
            if len(self.args.device_ids) > 0:
                inputs = inputs.cuda(self.args.device_ids[0])
                outputs = outputs.cuda(self.args.device_ids[0])
            _, _, _, f_1_gpu = self(inputs)
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

            # The number of validation itteration
            self.val_iter +=1 
            new_image.save(os.path.join(self.args.msd_results_root, "Training",
                                                    "Eval_Epoch: " + str(self.val_iter) +" Eval.png"))

                

        # tensorboard_logs = {'avg_val_loss': avg_loss}
        # # use key 'log'
        # return {'val_loss': avg_loss, 'log': tensorboard_logs}
        self.log('avg_val_loss', avg_loss)
        # self.logger.experiment.log('avg_val_loss', avg_loss)
    
