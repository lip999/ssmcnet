# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset
from metrics import dice_coef, iou_score, recall, precision, f2, hd_95
import losses
from utils import str2bool, count_params
import joblib
import imageio
import xlwt
from dataset import *
import unet_model

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='UNet_SSMCNet',help='model name')
    parser.add_argument('--mode', default='Calculate',help='GetPicture or Calculate')
    args = parser.parse_args()
    return args

def main():
    val_args = parse_args()

    for kk in range(5):
        kk = kk + 1
        result = r'D:\Pycharm\atlas_2D\models\ssmcnet/' + str(kk) + '_' + val_args.name
        args = joblib.load(os.path.join(result,"logs/args.pkl"))
        output = os.path.join(result, "output" + "/")
        if not os.path.exists(output):
            os.makedirs(output)
        print('Config -----')
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)))
        print('------------')

        joblib.dump(args,  result + '/' + 'logs/args.pkl')
        # create model
        print("=> creating model %s" % args.arch)

        model = unet_urpc.__dict__[args.arch](args)
        model = model.cuda()

        # Data loading code
        val_img_paths = r'D:\Pycharm\atlas_2D\data/' + 'data' + str(kk) + '/' + 'test/IMAGE/'
        val_mask_paths = r'D:\Pycharm\atlas_2D\data/' + 'data' + str(kk) + '/' + 'test/LABEL/'

        prepath = os.path.join(output, "pre_mask" + "/")  # pre_mask
        prepath1 = os.path.join(output, "pre_image" + "/")  # pre_image

        if not os.path.exists(prepath):
            os.mkdir(prepath)
        if not os.path.exists(prepath1):
            os.mkdir(prepath1)
      
        model.load_state_dict(torch.load(result + '/' +'models/bestmodel.pth'))
        model.eval()

        val_dataset =BasicDataset(val_img_paths, val_mask_paths, scale=1)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        if val_args.mode == "Calculate":
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with torch.no_grad():
                    for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                        input = input.cuda()
                        if args.deepsupervision:
                            output = model(input)[-1]
                        else:
                            output = model(input)
                        output = torch.sigmoid(output[0]).data.cpu().numpy()
                        img_paths = glob(val_img_paths + "*.npy")
                        img_paths = img_paths[args.batch_size * i:args.batch_size * (i + 1)]  # 根据batchsize的大小确定有几个样本
                        for i in range(output.shape[0]):  # i对应的batchsize的大小
            
                            npName = os.path.basename(img_paths[i])
                            overNum = npName.find(".npy")
                            rgbName = npName[0:overNum]
            
                            output_4 = np.squeeze(output, axis=1)  # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
                            maskImg = output_4[i, :, :]
                            maskpath1 = prepath1 + "\\" + rgbName
                            np.save(maskpath1, maskImg)  #pre_img
                            masknpy = np.zeros(np.shape(maskImg))
                            for h in range(maskImg.shape[0]):
                                for w in range(maskImg.shape[1]):
                                    if maskImg[h][w] >0.5:
                                        masknpy[h][w] = 1
                                    else:
                                        masknpy[h][w] = 0
                            rgbName = rgbName + ".npy"
                            maskpath = prepath + "\\" + rgbName
                            np.save(maskpath,masknpy)  #pre_mask
                torch.cuda.empty_cache()
            """
            计算各种指标:Dice、iou、、
            """
            wt_dices = []
            wt_recalls = []
            wt_precisions = []
            wt_Hausdorf = []
            wt_iou = []
            wt_F2 = []

            maskPath = glob(prepath + "*.npy")
            gtPath = glob(val_mask_paths+ "*.npy")

            wb = xlwt.Workbook(encoding='utf-8')
            ws = wb.add_sheet('评价指标')
            j = 1

            ws.write(0, 0, label="病人")
            ws.write(0, 1, label="wt_dices")
            ws.write(0, 2, label="wt_iou")
            ws.write(0, 3, label="wt_recalls")
            ws.write(0, 5, label="wt_precisions")
            ws.write(0, 6, label="wt_Hausdorf")
            ws.write(0, 7, label="wt_F2")


            for myi in tqdm(range(len(maskPath))):
                mask = np.load(maskPath[myi])
                gt = np.load(gtPath[myi])
                wtpbregion =mask
                wtmaskregion= gt

                npName1 = os.path.basename(maskPath[myi])
                overNum1 = npName1.find(".npy")
                Name = npName1[0:overNum1]


                dice = dice_coef(wtpbregion,wtmaskregion)
                wt_dices.append(dice)
                re= recall(wtpbregion,wtmaskregion)
                wt_recalls.append(re)
                pre=precision(wtpbregion,wtmaskregion)
                wt_precisions.append(pre)
                Hausdorff = hd_95(wtpbregion, wtmaskregion)
                wt_Hausdorf.append(Hausdorff)
                iou = iou_score(wtpbregion, wtmaskregion)
                wt_iou.append(iou)
                F2 = f2(wtpbregion, wtmaskregion)
                wt_F2.append(F2)


                ws.write(j, 0, label=Name)
                ws.write(j, 1, label=float(wt_dices[myi]))
                ws.write(j, 2, label=float(wt_iou[myi]))
                ws.write(j, 3, label=float(wt_recalls[myi]))
                ws.write(j, 4, label=float(wt_precisions[myi]))
                ws.write(j, 5, label=float(wt_Hausdorf[myi]))
                ws.write(j, 6, label=float(wt_F2[myi]))
                j=j+1

            ws.write(j + 1, 0, label="平均值")
            ws.write(j + 1, 1, label=np.mean(wt_dices))
            ws.write(j + 1, 2, label=np.mean(wt_iou))
            ws.write(j + 1, 3, label=np.mean(wt_recalls))
            ws.write(j + 1, 4, label=np.mean(wt_precisions))
            ws.write(j + 1, 5, label=np.mean(wt_Hausdorf))
            ws.write(j + 1, 6, label=np.mean(wt_F2))

            print('WT Dice: %.4f' % np.mean(wt_dices))
            print('WT iou: %.4f' % np.mean(wt_iou))
            print('WT recalls: %.4f' % np.mean(wt_recalls))
            print('WT precisions: %.4f' % np.mean(wt_precisions))
            print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
            print('WT f2: %.4f' % np.mean(wt_F2))

            xls = os.path.join(output + "/" + "metric.xls")
            wb.save(xls)

            print("Done!")


        if val_args.mode == "GetPicture":
            prepngpath = os.path.join(output, "prepng" + "/")
            gtpngpath = os.path.join(output, "gtpng" + "/")

            if not os.path.exists(prepngpath):
                os.mkdir(prepngpath)
            if not os.path.exists(gtpngpath):
                os.mkdir(gtpngpath)
            masks=os.listdir(prepath)
            for mask in masks:
                maskpath=os.path.join(prepath, mask)
                mask_npy=np.load(maskpath)
                # mask_npy = np.array(mask_npy)
                maskImg = (mask_npy * 255).astype(np.uint8)
                maskname = mask.split('.')[0]
                imsave(prepngpath + str(maskname) + ".png", maskImg)

            gts = os.listdir(val_mask_paths)
            for gt in gts:
                gtpath = os.path.join(val_mask_paths, gt)
                gt_npy = np.load(gtpath)
                # gt_npy = np.array(gt_npy)
                gtImg = (gt_npy * 255).astype(np.uint8)
                gtname = gt.split('.')[0]
                imsave(gtpngpath + str(gtname) + ".png", gtImg)
            print("Done!")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.enabled = False
    main( )
