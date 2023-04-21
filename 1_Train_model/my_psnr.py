

from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob


from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter


VAL_DIR='F:\KEY\FINAL\example\MY-SR\\val'

            # Test for validation images
files_gt = glob.glob(VAL_DIR + '/HR/*.jpg')
files_gt.sort()
files_lr = glob.glob(VAL_DIR + '/LR/*.jpg')
files_lr.sort()

psnrs_hr = []
psnrs_lr=[]
lpips = []

for ti, fn in enumerate(files_gt):
                # Load HR image
        tmp_hr = _load_img_array(files_gt[ti])
        val_H = np.asarray(tmp_hr).astype(np.float32)  # HxWxC

                # Load LR image
        tmp_lr = _load_img_array(files_lr[ti])
        val_L = np.asarray(tmp_lr).astype(np.float32)  # HxWxC
        val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
        val_L = val_L[np.newaxis, ...]            # BxCxHxW


                # PSNR on Y channel
        img_gt = (val_H*255).astype(np.uint8)
                #print(image_out.shape)
        CROP_S = 4
        psnrs_hr.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(tmp_hr)[:,:,0], CROP_S))
        psnrs_lr.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(tmp_lr)[:,:,0], CROP_S))


print('HR AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs_hr))))
print('LR AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs_lr))))
