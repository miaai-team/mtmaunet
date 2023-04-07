import numpy as np
import torch
from monai.metrics import ROCAUCMetric, DiceMetric, HausdorffDistanceMetric,compute_dice
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete, Compose, EnsureType, Spacing
from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score,recall_score

        
class Evaluator_Seg(object):
    def __init__(self,num_class, include_background=False, reduction="mean",percentile=95,distance_metric = "euclidean"):
        self.dice_metric = DiceMetric(include_background, reduction=reduction)
        self.hd95_metric = HausdorffDistanceMetric(include_background, reduction=reduction,percentile=percentile,distance_metric=distance_metric)
        self.y_pred_trans = Compose([Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=num_class)])
        self.y_trans = Compose([AsDiscrete(to_onehot=num_class)])
        self.y_pred_trans_hd = Compose([Spacing(pixdim=[1,1,1],mode='nearest'), Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=num_class)])
        self.y_trans_hd = Compose([Spacing(pixdim=[1,1,1],mode='nearest'),   AsDiscrete(to_onehot=num_class)])

    def Dice(self,reduction=None):
        return self.dice_metric.aggregate(reduction)

    def HD95(self,reduction=None):
        return self.hd95_metric.aggregate(reduction)
        
    def add_batch(self, y_pred, y):
        y_pred_onehot = [self.y_pred_trans(i) for i in decollate_batch(y_pred)]
        y_onehot = [self.y_trans(i) for i in decollate_batch(y)]
        self.dice_metric(y_pred_onehot, y_onehot)

        y_pred_onehot_hd = [self.y_pred_trans_hd(i) for i in decollate_batch(y_pred)]
        y_onehot_hd = [self.y_trans_hd(i) for i in decollate_batch(y)]
        self.hd95_metric(y_pred_onehot_hd, y_onehot_hd)
        
    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()
    

class Evaluator_Cls(object):
    def __init__(self,num_class):
        self.auc_metric = ROCAUCMetric()
        self.y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
        self.y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])
        self.argmax_trans = AsDiscrete(argmax=True)
        self.y_pred = []
        self.y = []
        self.y_onehot = []
        self.y_pred_act = []

    def to_one_hot(self):
        if self.y_onehot == [] and self.y_pred_act == []:
            y_pred = torch.cat(self.y_pred,dim=0)
            y = torch.cat(self.y,dim=0)
            self.y_onehot = [self.y_trans(i).as_tensor().cpu() for i in decollate_batch(y)]
            self.y_pred_act = [self.y_pred_trans(i).as_tensor().cpu() for i in decollate_batch(y_pred)]

    def to_class(self):
        self.to_one_hot()
        self.y_class = [self.argmax_trans(i).item() for i in  self.y_onehot]
        self.y_pred_class = [self.argmax_trans(i).item() for i in self.y_pred_act]

    def AUC(self):
        self.to_one_hot()
        self.auc_metric(self.y_pred_act, self.y_onehot)
        return self.auc_metric.aggregate()
    
    def ACC(self):
        self.to_class()
        score = accuracy_score(self.y_class,self.y_pred_class)
        return score

    def F1(self,average='macro'):
        self.to_class()
        score = f1_score(self.y_class,self.y_pred_class,average=average)
        return score

    def Recall(self, labels=None,pos_label=1,average="binary",):
        self.to_class()
        score = recall_score(self.y_class,self.y_pred_class,labels=labels,pos_label=pos_label, average=average)
        return score

    def Report(self):
        report = classification_report(self.y_class,self.y_pred_class)
        return report
    
    def Confusion_matrix(self):
        cm = confusion_matrix(self.y_class,self.y_pred_class)
        return cm

    def Performance(self, y_pred, y):
        y_class = [self.argmax_trans(self.y_trans(i)) for i in decollate_batch(y)][0].as_tensor().cpu()
        y_pred_class = [self.argmax_trans(self.y_pred_trans(i)) for i in decollate_batch(y_pred)][0].as_tensor().cpu()
        score = accuracy_score(y_class,y_pred_class)
        return score,y_pred_class.item(),y_class.item()
        
    def add_batch(self, y_pred, y):
        self.y_pred.append(y_pred.detach())
        self.y.append(y.detach())
        
    def reset(self):
        self.y_pred=[]
        self.y=[]
        self.y_onehot = []
        self.y_pred_act = []
        self.auc_metric.reset()


class Evaluator_Seg_Infer(object):
    def __init__(self, include_background=False, reduction="mean",percentile=95,distance_metric = "euclidean"):       
        self.dice_metric = DiceMetric(include_background, reduction=reduction)
        self.hd95_metric = HausdorffDistanceMetric(include_background, reduction=reduction,percentile=percentile,distance_metric=distance_metric)
        self.y_pred_trans_hd = Compose([Spacing(pixdim=[1,1,1],mode='nearest')])
        self.y_trans_hd = Compose([Spacing(pixdim=[1,1,1],mode='nearest')])


    def Dice(self, y_pred, y,reduction=None):
        """ y and pred  is list  """
        self.dice_metric(y_pred, y)
        dice = self.dice_metric.aggregate(reduction)
        self.dice_metric.reset()
        return dice

    def HD95(self,y_pred, y,reduction=None):
        """ y and pred  is list  """
        y_pred_onehot_hd = [self.y_pred_trans_hd(i) for i in  y_pred]
        y_onehot_hd = [self.y_trans_hd(i) for i in  y]
        self.hd95_metric(y_pred_onehot_hd, y_onehot_hd)
        hd95 = self.hd95_metric.aggregate(reduction)
        self.hd95_metric.reset()
        return hd95


class Evaluator_Cls_Infer(object):
    def __init__(self,num_class):
        self.y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
        self.y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])
        self.y_pred = []
        self.y = []
        self.y_onehot = []
        self.y_pred_act = []

    def to_one_hot(self):
        if self.y_onehot == [] and self.y_pred_act == []:
            y_pred = torch.cat(self.y_pred,dim=0)
            y = torch.cat(self.y,dim=0)
            self.y_onehot = [self.y_trans(i) for i in decollate_batch(y)]
            self.y_pred_act = [self.y_pred_trans(i) for i in decollate_batch(y_pred)]

    def to_class(self):
        self.to_one_hot()
        self.y_class = [ np.argmax(i.cpu().numpy()).item() for i in  self.y_onehot]
        self.y_pred_class = [ np.argmax(i.cpu().numpy()).item() for i in self.y_pred_act]

    
    def Performance(self, y_pred, y):
        self.y_pred.append(y_pred.detach())
        self.y.append(y.detach())
        self.to_class()
        score = accuracy_score(self.y_class,self.y_pred_class)
        y = self.y_class
        y_pred = self.y_pred_class
        self.reset()
        return score,y_pred,y
        
    def reset(self):
        self.y_pred=[]
        self.y=[]
        self.y_onehot = []
        self.y_pred_act = []

