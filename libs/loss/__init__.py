# -*- coding:utf-8 -*- #
import yaml
import logging
from addict import Dict
import torch.nn as nn
from .loss import BoundaryLoss, GeneralizedCELoss, AutomaticWeightedLoss, FocalLoss_cls,SampleWeightedCELoss
# from .loss import FocalLoss,DiceLoss,DistanceLoss, BoundaryLoss
# from .loss import FocalLoss,DiceLoss,DistanceLoss, BoundaryLoss

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss, TverskyLoss


key2opt = {
    "focal": FocalLoss,
    "crossentropy": GeneralizedCELoss,
    'dice':DiceLoss,
    'dicece':DiceCELoss,
    'dicefocal':DiceFocalLoss,
    'tversky':TverskyLoss,
    'boundary': BoundaryLoss
}


class CompoundedLoss(nn.Module):
    def __init__(self, cfg):
        super(CompoundedLoss, self).__init__()
        self.cfg = cfg
        self._get_losses()

    def _get_losses(self):
        self.losses = []
        self.loss_w = []
        if self.cfg.solver.loss is None:
            self.losses.append(get_single_lossfunction()())
            self.loss_w.append(1)
            return
        for loss_type in self.cfg.solver.loss:
            loss_cls = get_single_lossfunction(loss_type)
            try:
                loss_params = {k: v for k, v in self.cfg['solver']['loss'][loss_type].items() if k != 'self_weight'}
                self.losses.append(loss_cls(**loss_params))
            except:
                self.losses.append(loss_cls())
            try:
                self.loss_w.append(int(self.cfg['solver']['loss'][loss_type]['self_weight']))
            except:
                self.loss_w.append(1)

    def forward(self, inputs, targets, dist_maps=None):
        all_loss = 0
        for i, loss in enumerate(self.losses):
            if isinstance(loss, BoundaryLoss):
                assert dist_maps is not None and dist_maps.shape == inputs.shape, \
                    ' dist maps  is required and has the same dimension as the inputs when using Boundary Loss\n' \
                    'You may need to add a dist maps during the dataset load'
                all_loss += self.loss_w[i] * loss(inputs, dist_maps)
            else:
                all_loss += self.loss_w[i] * loss(inputs, targets)
        return all_loss


def get_single_lossfunction(loss_name=None):
    if loss_name is None:
        print("Using GeneralizedCELoss")
        return GeneralizedCELoss
    else:
        if loss_name not in key2opt:
            raise NotImplementedError("Loss type {} not implemented".format(loss_name))
        print("Add {} optimizer".format(loss_name))
        return key2opt[loss_name]


def get_lossfunction(cfg):
        return CompoundedLoss(cfg)
