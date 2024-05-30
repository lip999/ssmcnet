# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leakyrelu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x

def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x

class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x

class SUBlock(nn.Module):
    def __init__(self, in_channels, dropout_p):
        super(SUBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, xj, xk):
        xjk = F.relu(self.conv1(torch.cat([xj, xk], dim=1)))
        Fjk = (xj * xjk) + (xk * xjk)
        DFjk = self.conv2(abs(xj - Fjk))

        return DFjk

class Decoder_SSMC_MSNet(nn.Module):
    def __init__(self, params):
        super(Decoder_SSMC_MSNet, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.transconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 原来是kernel_size=4，padding=1
        self.transconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.transconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.transconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.SU1 = SUBlock(self.ft_chns[0], dropout_p=0.0)
        self.SU2 = SUBlock(self.ft_chns[1], dropout_p=0.0)
        self.SU3 = SUBlock(self.ft_chns[2], dropout_p=0.0)
        self.SU4 = SUBlock(self.ft_chns[3], dropout_p=0.0)

        self.conv1 = nn.Conv2d(self.ft_chns[0] * 4, self.ft_chns[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.ft_chns[1] * 3, self.ft_chns[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.ft_chns[2] * 2, self.ft_chns[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.ft_chns[3], self.ft_chns[3], kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)   #3, padding=1
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class, kernel_size=1)   #3, padding=1
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class, kernel_size=1)   #3, padding=1
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class, kernel_size=1)   #3, padding=1
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]
        x5 = feature[4]

        # 多尺度减法的融合，所有层累加
        x5_up = self.transconv5(x5)
        x5_up_up = self.transconv4(x5_up)
        x5_up_up_up = self.transconv3(x5_up_up)
        x5_up_up_up_up = self.transconv2(x5_up_up_up)
        x4_up = self.transconv4(x4)
        x4_up_up = self.transconv3(x4_up)
        x4_up_up_up = self.transconv2(x4_up_up)
        x3_up = self.transconv3(x3)
        x3_up_up = self.transconv2(x3_up)
        x2_up = self.transconv2(x2)

        # SU
        df12 = self.SU1(x1, x2_up)
        df13 = self.SU1(x1, x3_up_up)
        df14 = self.SU1(x1, x4_up_up_up)
        df15 = self.SU1(x1, x5_up_up_up_up)
        df23 = self.SU2(x2, x3_up)
        df24 = self.SU2(x2, x4_up_up)
        df25 = self.SU2(x2, x5_up_up_up)
        df34 = self.SU3(x3, x4_up)
        df35 = self.SU3(x3, x5_up_up)
        df45 = self.SU4(x4, x5_up)

        # MFFU
        dfc1 = self.conv1(torch.cat([df12, df13, df14, df15], dim=1))
        avg_pool1 = torch.mean(dfc1, dim=1, keepdim=True)
        df1 = dfc1 * (self.sigmoid(self.relu(avg_pool1)))
        f1 = df1 + x1

        dfc2 = self.conv2(torch.cat([df23, df24, df25], dim=1))
        avg_pool2 = torch.mean(dfc2, dim=1, keepdim=True)
        df2 = dfc2 * (self.sigmoid(self.relu(avg_pool2)))
        f2 = df2 + x2

        dfc3 = self.conv3(torch.cat([df34, df35], dim=1))
        avg_pool3 = torch.mean(dfc3, dim=1, keepdim=True)
        df3 = dfc3 * (self.sigmoid(self.relu(avg_pool3)))
        f3 = df3 + x3

        dfc4 = self.conv4(torch.cat([df45], dim=1))
        avg_pool4 = torch.mean(dfc4, dim=1, keepdim=True)
        df4 = dfc4 * (self.sigmoid(self.relu(avg_pool4)))
        f4 = df4 + x4

        x = self.up1(x5, f4)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg_up = torch.nn.functional.interpolate(dp3_out_seg, shape)
        dp3_out_Dropout = self.out_conv_dp3(Dropout(x, p=0.5))

        x = self.up2(x, f3)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg_up = torch.nn.functional.interpolate(dp2_out_seg, shape)
        dp2_out_FeatureDropout = self.out_conv_dp2(FeatureDropout(x))

        x = self.up3(x, f2)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg_up = torch.nn.functional.interpolate(dp1_out_seg, shape)
        dp1_out_feature_noise = self.out_conv_dp1(self.feature_noise(x))

        x = self.up4(x, f1)
        dp0_out_seg = self.out_conv(x)
        dp0_out_Dropout = self.out_conv(Dropout(x, p=0.5))
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, \
               dp1_out_seg_up, dp2_out_seg_up, dp3_out_seg_up, \
               dp0_out_Dropout, dp1_out_feature_noise, dp2_out_FeatureDropout, dp3_out_Dropout

class UNet_SSMCNet(nn.Module):
    def __init__(self, args):
        super(UNet_SSMCNet, self).__init__()

        params = {'in_chns': 1,
                  'feature_chns': [64, 128, 256, 512, 1024],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': 1,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_SSMC_MSNet(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, \
        dp1_out_seg_up, dp2_out_seg_up, dp3_out_seg_up, \
        dp0_out_Dropout, dp1_out_feature_noise, dp2_out_FeatureDropout, dp3_out_Dropout = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg, \
               dp1_out_seg_up, dp2_out_seg_up, dp3_out_seg_up, \
               dp0_out_Dropout, dp1_out_feature_noise, dp2_out_FeatureDropout, dp3_out_Dropout


