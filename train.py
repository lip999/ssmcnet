# -*- coding: utf-8 -*-
import os
import time
import argparse
from collections import OrderedDict
from tqdm import tqdm
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from metrics import iou_score
import losses
from utils import str2bool, all_params
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader, random_split
import ramps
import unet_model

arch_names = list(unet_model.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNet_SSMCNet',choices=arch_names,
                        help='model architecture: ' + ' | '.join(arch_names) + ' (default: NestedUNet)')          
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 12)')
    parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--loss', default='BCE_loss', choices=loss_names, help='loss: ' + ' | '.join(loss_names) + ' (default:BCEDiceLoss)')

    parser.add_argument('--input-channels', default=1, type=int,  help='input channels')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +' | '.join(['Adam', 'SGD']) +' (default: Adam)')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # costs
    parser.add_argument('--consistency', type=float,default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,default=200.0, help='consistency_rampup')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value 计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_current_consistency_weight(args,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def Distance(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='none')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    return (KLDivLoss(q_output, p_output) + KLDivLoss(p_output, q_output))/2

def semi_supervised_loss(args,outputs0, outputs1, outputs2, outputs3, outputs1_up, outputs2_up, outputs3_up,outputs0_noise, outputs1_noise, outputs2_noise, outputs3_noise, target, criterion, iter_num):
    # 监督学习
    loss0 = criterion(outputs0[:args.labeled_bs], target[:args.labeled_bs])
    loss1 = criterion(outputs1_up[:args.labeled_bs], target[:args.labeled_bs])
    loss2 = criterion(outputs2_up[:args.labeled_bs], target[:args.labeled_bs])
    loss3 = criterion(outputs3_up[:args.labeled_bs], target[:args.labeled_bs])
    supervised_loss = (loss0 + loss1 + loss2 + loss3) / 4
    
    # 一致性学习
    outputs0_soft = torch.softmax(outputs0, dim=1)
    outputs1_soft = torch.softmax(outputs1, dim=1)
    outputs2_soft = torch.softmax(outputs2, dim=1)
    outputs3_soft = torch.softmax(outputs3, dim=1)

    outputs0_noise_soft = torch.softmax(outputs0_noise, dim=1)
    outputs1_noise_soft = torch.softmax(outputs1_noise, dim=1)
    outputs2_noise_soft = torch.softmax(outputs2_noise, dim=1)
    outputs3_noise_soft = torch.softmax(outputs3_noise, dim=1)

    loss_dis0 = torch.sum(Distance(torch.log(outputs0_soft[args.labeled_bs:]), outputs0_noise_soft[args.labeled_bs:]), dim=1, keepdim=True)
    loss_dis1 = torch.sum(Distance(torch.log(outputs1_soft[args.labeled_bs:]), outputs1_noise_soft[args.labeled_bs:]), dim=1, keepdim=True)
    loss_dis2 = torch.sum(Distance(torch.log(outputs2_soft[args.labeled_bs:]), outputs2_noise_soft[args.labeled_bs:]), dim=1, keepdim=True)
    loss_dis3 = torch.sum(Distance(torch.log(outputs3_soft[args.labeled_bs:]), outputs3_noise_soft[args.labeled_bs:]), dim=1, keepdim=True)

    if iter_num<5:
        consistency_weight = 1e-8
    else:
        consistency_weight = get_current_consistency_weight(args, iter_num // 5)

    consistency_MSE0 = (outputs0_noise_soft[args.labeled_bs:] - outputs0_soft[args.labeled_bs:]) ** 2
    exp_variance_main = torch.exp(-consistency_MSE0)
    consistency_loss0 = torch.mean(consistency_MSE0 * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(loss_dis0)
    consistency_MSE1 = (outputs1_noise_soft[args.labeled_bs:] - outputs1_soft[args.labeled_bs:]) ** 2
    exp_variance_aux1 = torch.exp(-consistency_MSE1)
    consistency_loss1 = torch.mean(consistency_MSE1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(loss_dis1)
    consistency_MSE2 = (outputs2_noise_soft[args.labeled_bs:] - outputs2_soft[args.labeled_bs:]) ** 2
    exp_variance_aux2 = torch.exp(-consistency_MSE2)
    consistency_loss2 = torch.mean(consistency_MSE2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(loss_dis2)
    consistency_MSE3 = (outputs3_noise_soft[args.labeled_bs:] - outputs3_soft[args.labeled_bs:]) ** 2
    exp_variance_aux3 = torch.exp(-consistency_MSE3)
    consistency_loss3 = torch.mean(consistency_MSE3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(loss_dis3)

    w0 = torch.exp(- consistency_loss0 / (consistency_loss0 + consistency_loss1 + consistency_loss2 + consistency_loss3 + 1e-8))
    w1 = torch.exp(- consistency_loss1 / (consistency_loss0 + consistency_loss1 + consistency_loss2 + consistency_loss3 + 1e-8))
    w2 = torch.exp(- consistency_loss2 / (consistency_loss0 + consistency_loss1 + consistency_loss2 + consistency_loss3 + 1e-8))
    w3 = torch.exp(- consistency_loss3 / (consistency_loss0 + consistency_loss1 + consistency_loss2 + consistency_loss3 + 1e-8))
    consistency_loss = consistency_loss0 * w0 + consistency_loss1 * w1 + consistency_loss2 * w2 + consistency_loss3 * w3

    loss = supervised_loss + consistency_weight * consistency_loss

    return loss

def train(args, train_loader, model, criterion, optimizer, iter_num, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        input = input.cuda()
        target = target.cuda()

        outputs0, outputs1, outputs2, outputs3, outputs1_up, outputs2_up, outputs3_up, \
        outputs0_noise, outputs1_noise, outputs2_noise, outputs3_noise = model(input)
        
        loss = semi_supervised_loss(args, outputs0, outputs1, outputs2, outputs3,outputs1_up, outputs2_up, outputs3_up,
                                    outputs0_noise, outputs1_noise, outputs2_noise, outputs3_noise,target, criterion, iter_num)

        iou0 = iou_score(outputs0, target)
        iou1 = iou_score(outputs1_up, target)
        iou2 = iou_score(outputs2_up, target)
        iou3 = iou_score(outputs3_up, target)
        iou = (iou0 + iou1 + iou2 + iou3)/4

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, val_loader, model, criterion, optimizer, iter_num):
    losses = AverageMeter()
    ious = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output[0], target)
            iou = iou_score(output[0], target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log

def main():
    args = parse_args()

    for kk in range(5):
        kk = kk + 1
        if args.name is None:
            args.name = '%s_%s' %(args.arch,args.loss)

        result = r'D:\Pycharm\atlas_2D\models\ssmcnet/' + str(kk) + '_' + args.name
        models=os.path.join(result,"models"+"/")
        logs=os.path.join(result,"logs"+"/")

        img_paths = r'D:\Pycharm\atlas_2D\data/' + 'data' + str(kk) + '/' + 'train/IMAGE/'
        mask_paths = r'D:\Pycharm\atlas_2D\data/' + 'data' + str(kk) + '/' + 'train/LABEL/'

        if not os.path.exists(result):
            os.makedirs(result)
        if not os.path.exists(models):
            os.makedirs(models)
        if not os.path.exists(logs):
            os.makedirs(logs)

        print('Config -----')
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)))
        print('------------')

        with open(logs+"/"+"args.txt", 'w') as f:
            for arg in vars(args):
                print('%s: %s' %(arg, getattr(args, arg)), file=f)

        joblib.dump(args, logs+"/"+'args.pkl' )

        if args.loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion = losses.__dict__[args.loss]().cuda()
        cudnn.benchmark = True

        print("=> creating model %s" %args.arch)
        model = unet_model.__dict__[args.arch](args)
        model = model.cuda()
        model.apply(weigth_init)

        print("所有参数：",all_params(model))

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        dataset = BasicDataset(img_paths,  mask_paths, scale=1)
        n_val = int(len(dataset) * 0.25)
        n_train = len(dataset) - n_val
        print("n_val.num",n_val)
        print("n_train.num",n_train)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        total_slices = len(train_dataset)
        label_slice = int(len(train_dataset) * 0.5)
        print("Total slices is: {}, labeled slices is: {}".format(total_slices, label_slice))
        labeled_idxs = list(range(0, label_slice))
        unlabeled_idxs = list(range(label_slice, total_slices))
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

        with open(logs + "/" + "params.txt", 'w') as f:
            print('train params: ', all_params(model) , file=f)
            print("n_train.num: ", n_train, file=f)
            print("n_val.num: ", n_val, file=f)
            print("label slices: ", label_slice, file=f)
            print("Total slices: ", total_slices, file=f)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            pin_memory=True,
            num_workers=8)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=8)

        log = pd.DataFrame(index=[], columns=[ 'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])
        best_loss = 10
        trigger = 0
        iter_num = 0
        torch.cuda.synchronize()
        time_first = time.time()
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' %(epoch, args.epochs))
            torch.cuda.synchronize()
            time_start = time.time()
            train_log = train(args, train_loader, model, criterion, optimizer, iter_num, epoch)
            val_log = validate(args, val_loader, model, criterion, optimizer, iter_num)
            time_end = time.time()
            runtime = time_end - time_start
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - runtime %.4f'
                  %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], runtime))
            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                train_log['iou'],
                val_log['loss'],
                val_log['iou'],
                runtime
            ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou', 'runtime'])
            log = log.append(tmp, ignore_index=True)
            log.to_csv(logs+"/"+'log.csv', index=False)
            trigger += 1
            iter_num = iter_num + 1

            if val_log['loss'] < best_loss:
                a=str(best_loss-val_log['loss'])
                torch.save(model.state_dict(), models+"/"+'bestmodel.pth')
                best_loss = val_log['loss']
                print("=> saved best model")
                if a > str(0.0001):
                    trigger = 0
            print(trigger)

            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping")
                    torch.cuda.synchronize()
                    with open(logs + "/" + "params.txt", 'a+') as f:
                        print("total runtime ", time.time() - time_first, file=f)
                    break

            torch.cuda.empty_cache()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    main()
