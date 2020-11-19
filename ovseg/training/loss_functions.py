import torch
from torch import nn
import numpy as np


class cross_entropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logs, yb_oh):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        return self.loss(logs, yb_int)


class dice_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logs, yb_oh):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        fn = torch.sum(yb_oh * (1 - pred), dim)
        fp = torch.sum((1 - yb_oh) * pred, dim)
        # now the dice score, excited?
        dice = (2 * tp + self.eps) / (2*tp + fp + fn + self.eps)
        return -1 * dice.mean()


def to_one_hot_encoding(logs, yb):

    yb = yb.long()
    n_ch = logs.shape[1]
    yb_oh = torch.cat([(yb == c) for c in range(n_ch)], 1).float()
    return yb_oh


class CE_dice_loss(nn.Module):

    def __init__(self, eps=1e-5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = cross_entropy()
        self.dice_loss = dice_loss(eps)
        self.dice_weight = 0.5
        self.ce_weight = 1 - self.dice_weight

    def forward(self, logs, yb):
        if yb.shape[1] == 1:
            # turn yb to one hot encoding
            yb = to_one_hot_encoding(logs, yb)
        ce = self.ce_loss(logs, yb) * self.ce_weight
        dice = self.dice_loss(logs, yb) * self.dice_weight
        loss = ce + dice
        return loss


def downsample_yb(logs_list, yb):
    is_3d = len(yb.shape) == 5
    yb_list = [yb]
    for logs in logs_list[1:]:
        # maybe downsample in x direction
        if logs.shape[2] == yb.shape[2]:
            continue
        elif logs.shape[2] == yb.shape[2] // 2:
            yb = (yb[:, :, ::2] + yb[:, :, 1::2]) / 2
        else:
            raise ValueError('shapes of logs and labels aren\'t machting for '
                             'downsampling. got {} and {}'
                             .format(logs.shape, yb.shape))
        # maybe downsample in y direction
        if logs.shape[3] == yb.shape[3]:
            continue
        elif logs.shape[3] == yb.shape[3] // 2:
            yb = (yb[:, :, :, ::2] + yb[:, :, :, 1::2]) / 2
        else:
            raise ValueError('shapes of logs and labels aren\'t machting for '
                             'downsampling. got {} and {}'
                             .format(logs.shape, yb.shape))
        # maybe downsample in z direction
        if is_3d:
            if logs.shape[4] == yb.shape[4]:
                continue
            elif logs.shape[4] == yb.shape[4] // 2:
                yb = (yb[:, :, :, :, ::2] + yb[:, :, :, :, 1::2]) / 2
            else:
                raise ValueError('shapes of logs and labels aren\'t machting '
                                 'for downsampling. got {} and {}'
                                 .format(logs.shape, yb.shape))
        # now append
        yb_list.append(yb)
    return yb_list


class CE_dice_pyramid_loss(nn.Module):

    def __init__(self, eps=1e-5, dice_weight=0.5, pyramid_weight=0.5):
        super().__init__()
        self.ce_dice_loss = CE_dice_loss(eps, dice_weight)
        self.pyramid_weight = pyramid_weight

    def forward(self, logs_list, yb):
        # compute the weights to be powers of pyramid_weight
        scale_weights = self.pyramid_weight ** np.arange(len(logs_list))
        # let them sum to one
        scale_weights = scale_weights / np.sum(scale_weights)
        # turn labels into one hot encoding and downsample to same resolutions
        # as the logits
        yb = to_one_hot_encoding(logs_list[0], yb)
        yb_list = downsample_yb(logs_list, yb)

        # now let's compute the loss for each scale
        loss = 0
        for logs, yb, w in zip(logs_list, yb_list, scale_weights):
            loss += w * self.ce_dice_loss(logs, yb)

        return loss
