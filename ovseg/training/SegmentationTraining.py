from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss, to_one_hot_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationTraining(NetworkTraining):

    def __init__(self, *args,
                 prg_trn_sizes=None,
                 prg_trn_arch_params=None,
                 prg_trn_aug_params=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prg_trn_sizes = prg_trn_sizes
        self.prg_trn_arch_params = prg_trn_arch_params
        self.prg_trn_aug_params = prg_trn_aug_params

        # now have fun with progressive training!
        self.do_prg_trn = self.prg_trn_sizes is not None
        if self.do_prg_trn:
            self.prg_trn_n_stages = len(self.prg_trn_sizes)
            assert self.prg_trn_n_stages > 1, "please use progressive training only if you have "\
                "more then one stage."
            self.prg_trn_epochs_per_stage = self.num_epochs // self.prg_trn_n_stages
            self.prg_trn_update_parameters()
        else:
            self.prg_trn_process_batch = identity()

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def compute_batch_loss(self, batch):

        batch = batch.cuda()
        if self.augmentation is not None:
            batch = self.augmentation(batch)

        xb, yb = batch[:, :-1], batch[:, -1:]
        yb = to_one_hot_encoding(yb, self.network.out_channels)
        xb, yb = self.prg_trn_process_batch(xb, yb)
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss

    def prg_trn_update_parameters(self):

        if self.epochs_done == self.num_epochs:
            return

        # compute which stage we are in atm
        self.prg_trn_stage = self.epochs_done // self.prg_trn_epochs_per_stage

        self.print_and_log('\nProgressive Training: '
                           'Stage {}, size {}'.format(self.prg_trn_stage,
                                                      self.prg_trn_sizes[self.prg_trn_stage]),
                           2)
        if self.prg_trn_stage < self.prg_trn_n_stages - 1:
            # the most imporant part of progressive training: we update the resizing function
            # that should make the batches smaller
            self.prg_trn_process_batch = resize(self.prg_trn_sizes[self.prg_trn_stage],
                                                self.network.is_2d)
        else:
            # here we assume that the last stage of the progressive training has the desired size
            # i.e. the size that the augmentation/the dataloader returns
            self.prg_trn_process_batch = identity()

        if self.prg_trn_arch_params is not None:
            # here we update architectural paramters, this should be dropout and stochastic depth
            # rate
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            self.network.update_prg_trn(self.prg_trn_arch_params, h)

        if self.prg_trn_aug_params is not None:
            # here we update augmentation parameters. The idea is we augment more towards the
            # end of the training
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            if self.augmentation is not None:
                self.augmentation.update_prg_trn(self.prg_trn_aug_params, h)
            if self.trn_dl.dataset.augmentation is not None:
                self.trn_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h)
            if self.val_dl is not None:
                if self.val_dl.dataset.augmentation is not None:
                    self.val_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h)

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.do_prg_trn:
            # if we do progressive training we update the parameters....
            if self.epochs_done % self.prg_trn_epochs_per_stage == 0:
                # ... if this is the right epoch for this
                self.prg_trn_update_parameters()


class identity(nn.Identity):

    def forward(self, xb, yb):
        return xb, yb


class resize(nn.Module):

    def __init__(self, size, is_2d):
        super().__init__()

        self.size = size
        self.is_2d = is_2d
        self.mode = 'bilinear' if self.is_2d else 'trilinear'

    def forward(self, xb, yb):
        xb_ch = xb.shape[1]
        batch = torch.cat([xb, yb], 1)
        batch = F.interpolate(batch, size=self.size, mode=self.mode)
        return batch[:, :xb_ch], batch[:, xb_ch:]
