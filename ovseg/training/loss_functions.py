import torch
from torch import nn
import numpy as np


class cross_entropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()

class dice_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the dice will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        fn = torch.sum(yb_oh * (1 - pred), dim)
        fp = torch.sum((1 - yb_oh) * pred, dim)
        # now the dice score, excited?
        dice = (2 * tp + self.eps) / (2*tp + fp + fn + self.eps)
        return -1 * dice.mean()


class cross_entropy_weighted_bg(nn.Module):

    def __init__(self, weight_bg, n_fg_classes):
        super().__init__()
        self.weight_bg = weight_bg
        self.n_fg_classes = n_fg_classes
        self.weight = [self.weight_bg] + [1] * self.n_fg_classes
        self.weight = torch.tensor(self.weight).type(torch.float)
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='none')
        
    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        yb_int = torch.argmax(yb_oh, 1)
        l = self.loss(logs, yb_int)
        if mask is not None:
            l = l * mask[:, 0]
        return l.mean()


class sens_loss(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logs, yb_oh, mask=None):
        assert logs.shape == yb_oh.shape
        pred = torch.nn.functional.softmax(logs, 1)
        # dimension in which we compute the mean
        dim = list(range(2, len(pred.shape)))
        # remove the background channel from both as the sens will only
        # be computed over foreground classes
        pred = pred[:, 1:]
        yb_oh = yb_oh[:, 1:]
        if mask is not None:
            pred = pred * mask
            # Is this second line neseccary? Probably not! But better be safe than sorry.
            yb_oh = yb_oh * mask
        # now compute the metrics
        tp = torch.sum(yb_oh * pred, dim)
        yb_vol = torch.sum(yb_oh, dim)
        # the main formula
        sens = tp / (yb_vol + self.eps)
        # the mean is computed over the batch and channel axis (excluding background)
        return 1 - 1 * sens.mean()
