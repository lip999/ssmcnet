import numpy as np

import torch
import torch.nn.functional as F
from mindspore.nn.metrics import HausdorffDistance

"""
output为预测结果，target为真实结果
"""


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def accuracy(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output = np.atleast_1d(output.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    tn = np.count_nonzero(~output & ~target)
    fp = np.count_nonzero(output & ~target)
    fn = np.count_nonzero(~output & target)
    tp = np.count_nonzero(output & target)

    try:
        acc = (tp + tn + smooth) / float(tp + tn + fn + fp + smooth)
    except ZeroDivisionError:
        acc = 0.0

    return acc


def recall(result, reference):
    smooth = 1e-5

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = (tp + smooth) / float(tp + fn +smooth)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def precision(result, reference):
    smooth = 1e-5

    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        precision = (tp +smooth) / float(tp + fp +smooth)
    except ZeroDivisionError:
        precision = 0.0

    return precision

def f2(result, reference):
    smooth = 1e-5
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)
    fn = np.count_nonzero(~result & reference)
    tp = np.count_nonzero(result & reference)

    try:
        f2_score = (5 * tp + smooth) / float(5 * tp + 4 * fn + fp + smooth)
    except ZeroDivisionError:
        f2_score = 0.0

    return f2_score

def hd_95(output, target):
    x = output
    y = target
    metric = HausdorffDistance()
    metric.clear()
    metric.update(x, y, 0)
    distance = metric.eval()

    return distance