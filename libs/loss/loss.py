import torch
import torch.nn as nn
import numpy as np


import torch.nn.functional as F
from torch.autograd import Variable
from torch import einsum

from torch import Tensor
from torch.nn.modules.loss import  CrossEntropyLoss


class GeneralizedCELoss(CrossEntropyLoss):
    """
    Compute Cross Entropy Loss, support one-hot
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        
        return F.cross_entropy(input, target, weight=self.weight,
                            ignore_index=self.ignore_index, reduction=self.reduction,
                            label_smoothing=self.label_smoothing)

class SampleWeightedCELoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None,ignore_index=-1,label_smoothing=0.0):
        super().__init__(weight,reduction='none')
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index 
        self.label_smoothing = label_smoothing
        self.reduction='none'

    def forward(self, input, target,sample_weight=None):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,
                                  weight=self.weight,ignore_index=self.ignore_index,
                                  label_smoothing=self.label_smoothing)
        if sample_weight is not None:
            ce_loss = sample_weight*ce_loss
        return ce_loss.mean()


class FocalLoss_cls(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,ignore_index=-1,reduction='mean',label_smoothing=0.0):
        super(FocalLoss_cls, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index 
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,
                                    weight=self.weight,ignore_index=self.ignore_index,
                                    label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class FocalLoss_seg(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_seg, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.Tensor(np.array(alpha)))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        targets = targets.long()
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(inputs.shape).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets[:, None, ...]
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[targets]

        probs = (P * class_mask).sum(1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


loss_l2 = nn.MSELoss()


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Arguments:
            input: (B, C, H, W)

        Return:
            loss
        """
        # (B, _, H, W) = input.shape
        targets = targets.long()
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(inputs.shape).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets[:,None,...]
        class_mask.scatter_(1, ids.data, 1.)
        true_positive = P*class_mask
        for i in range(true_positive.dim()-1,1,-1):
            true_positive=true_positive.sum(dim=i)
            P = P.sum(dim=i)
            class_mask = class_mask.sum(dim=i)
        dice_score = 2*true_positive/(P+class_mask)
        return 1 - dice_score.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, input, dist_maps):
        """
        Note:
            计算 boundary loss
            仅计算前景类的loss 请注意背景通道索引默认为 0
        Arguments:
            input: (B, C, ...)  N-D 维度
            dist_maps: (B, C, ...)  N-D 维度
                        预计算的距离图，距离图计算可调用：one_hot2dist/one_hot2dist_batch
       Return:
            loss
        """
        assert input.size() == dist_maps.size()
        input = F.softmax(input, dim=1)
        pc = input[:, 1:, ...].type(torch.float32)
        dc = dist_maps[:, 1:, ...].type(torch.float32)
        return einsum("...,...->...", pc, dc).mean()


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.zeros(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum

    def get_params(self):
        return torch.exp(-self.params).detach().cpu().numpy()



