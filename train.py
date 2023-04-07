import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import torch
import math
from addict import Dict

import torch.nn.functional as F
from time import time
from accelerate import Accelerator
from libs.optimizers import get_optimizer
from libs.models import get_network
from libs.loss import get_lossfunction, AutomaticWeightedLoss, FocalLoss_cls,SampleWeightedCELoss, DiceCELoss, DiceLoss
from libs.datasets.base import myDataset
from libs.datasets.split_data import split_dataset_with_cv
from libs.utils import saver, metric, LR_Scheduler,make_print_to_file
from tensorboardX import SummaryWriter
import datetime 

from sklearn.metrics import f1_score, confusion_matrix

from monai.inferers import sliding_window_inference
from monai.data import create_test_image_3d, list_data_collate, decollate_batch

from monai.losses import focal_loss
from monai.transforms import SpatialCrop,SpatialPad, Compose
from scipy.ndimage.measurements import center_of_mass

import torchvision.utils as vutils
import torchvision
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024*8, rlimit[1]))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Trainer(object):
    def __init__(self, config_path):
        config = Dict(yaml.load(open(config_path,'r'), Loader=yaml.FullLoader))
        self.args = config 
        
        ## Define accelerator
        accelerator_param = {k: v for k, v in config['exp']['accelerator'].items()}
        self.accelerator = Accelerator(**accelerator_param)
        self.device = self.accelerator.device
        ## Define Saver
        self.saver = saver.Saver(self.args, config_path)
        self.saver.save_experiment_config()

        ## Get confige
        self.dim = self.args.dataset.dim
        self.channel = self.args.dataset.channel
        self.n_classes = self.args.dataset.n_classes
        self.patch_size = self.args.dataset.patch_size
        
        ## Get Dataset arg
        assert self.args.dataset.cv.fold < self.args.dataset.cv.num, 'fold too big'
        
        fold_i_path = os.path.join(self.args.dataset.root,self.args.dataset.cv.dir_name,f'fold_{self.args.dataset.cv.fold}')
        train_csv_path = os.path.join(fold_i_path,self.args.dataset.split.train)
        val_csv_path = os.path.join(fold_i_path,self.args.dataset.split.val)
        
        ## Define Evaluator
        self.evaluator_seg_ctl = metric.Evaluator_Seg(self.n_classes,include_background=False, reduction="mean")
        self.evaluator_seg_jdm = metric.Evaluator_Seg(self.n_classes,include_background=False, reduction="mean")
        self.evaluator_cls = metric.Evaluator_Cls(2)
        # ## define dataloader of train and validation
        train_dataset = myDataset(
            root = self.args.dataset.root,
            csv_path = train_csv_path,
            no_channel= self.args.dataset.no_channel,
            patch_size= self.args.dataset.patch_size,
            is_train = True
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.solver.batch_size.train,
            num_workers=self.args.dataloader.num_workers,
            shuffle=True,
            collate_fn=list_data_collate,
        )
        
        val_dataset = myDataset(
            root = self.args.dataset.root,
            csv_path = val_csv_path,
            no_channel= self.args.dataset.no_channel,
            patch_size=  self.args.dataset.patch_size,
            is_train = False
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.args.solver.batch_size.test,
            num_workers=self.args.dataloader.num_workers,
            shuffle=False,
            collate_fn=list_data_collate,
        )
                     
        # Define network
        network_cls = get_network(config)
        network_param = {k: v for k, v in config['network'][config["network"]["type"]].items() if k != 'name'}
        self.model = network_cls(**network_param)
        
        # Define tensorboard
        self.writer = SummaryWriter(log_dir='runs/{}/fold_{}/{}/{}'.format(self.args.exp.id, self.args.dataset.cv.fold, self.saver.id,datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        
        # Define Criterion
        self.criterion_seg_ctl = DiceCELoss(include_background=True,
                                                to_onehot_y=True,
                                                sigmoid=False,
                                                softmax=True,
                                                lambda_dice=1.0,
                                                lambda_ce=1.0,)
        self.criterion_seg_jdm = DiceCELoss(include_background=True,
                                                to_onehot_y=True,
                                                sigmoid=False,
                                                softmax=True,
                                                lambda_dice=1.0,
                                                lambda_ce=1.0,)
        self.criterion_cls = SampleWeightedCELoss(ignore_index=-1,
                                                label_smoothing=0.1,  
                                                weight=torch.tensor([ 0.5, 0.5]).to(self.device)
                                                )
        self.criterion = AutomaticWeightedLoss(3)

        # Define Optimizer
        optimizer_cls = get_optimizer(config)
        optimizer_params = {k: v for k, v in config['solver']['optimizer'].items() if k != 'name'}
        optimizer_net = optimizer_cls(
            [{'params': self.model.parameters()},
             {'params': self.criterion.parameters(),'weight_decay': 0, 'lr':optimizer_params['lr']*10}],
            **optimizer_params)
        print("Using optimizer{}".format(optimizer_net))
        self.optimizer = optimizer_net

        # Define lr scheduler
        self.scheduler = LR_Scheduler(self.args.solver.lr_scheduler, self.args.solver.optimizer.lr,
                                            self.args.solver.epoch_max, len(self.train_loader),
                                            warmup_epochs=self.args.solver.warmup_epochs)
        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.init_model != 'None':
            if not os.path.isfile(self.args.init_model):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.init_model))
            checkpoint = torch.load(self.args.init_model)
            self.args.solver.epoch_start = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            if not self.args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.init_model, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.args.ft:
            self.args.solver.epoch_start = 0
            
        #  Device free
        self.criterion, self.model, self.optimizer = self.accelerator.prepare(self.criterion, self.model, self.optimizer)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True 
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        global n_iter

        for i, sample in enumerate(tbar):
            image = sample['img'].to(self.device)
            target = sample['seg'].to(self.device)
            target_jdm = sample['jdm'].to(self.device)
            target_cls = sample['label'].to(self.device)
            target_jdm_masked = sample['masked'].to(self.device)
            target_cls_weight = sample['weight'].to(self.device)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output, output_jdm, output_cls = self.model(image)

            loss_seg_ctl = self.criterion_seg_ctl(output, target)
            loss_seg_jdm = self.criterion_seg_jdm(output_jdm*target_jdm_masked, target_jdm)
            loss_cls = self.criterion_cls(output_cls, target_cls,target_cls_weight)
            loss = self.criterion(loss_seg_ctl,loss_seg_jdm,loss_cls)

            self.accelerator.backward(loss)
            self.optimizer.step()
            train_loss = loss.detach().item() + train_loss
            tbar.set_description(f'Train loss: {(train_loss / (i + 1)):.3f}')
            self.writer.add_scalar('training_loss', loss.detach().item(), n_iter)
            n_iter += 1
            if i ==0:
                image_show = image
                target_show = target
                output_show = output

        imgs_show = torchvision.utils.make_grid(image_show.as_tensor()[...,self.patch_size[-1]//2],normalize=True)
        masks_show = torchvision.utils.make_grid(target_show.as_tensor()[...,self.patch_size[-1]//2].float(),normalize=True)
        pred_show = torchvision.utils.make_grid(torch.argmax(output_show.as_tensor(),dim=1)[:,None,...,self.patch_size[-1]//2].float(),normalize=True)
        
        self.writer.add_image('mask/train',masks_show,epoch,dataformats='CHW')
        self.writer.add_image('mask_pred/train',pred_show,epoch,dataformats='CHW')
        self.writer.add_image('Img/train',imgs_show,epoch,dataformats='CHW')
        
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.solver.batch_size.train + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss/num_img_tr))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    @staticmethod
    def get_patch_img(image,mask,patch_size):
        assert image.shape[0]==1   # image shape 1,C,H,W,D
        assert mask.shape[0]==1 
        mask = torch.sum(mask,dim=1).cpu().numpy()
        mask[mask>0]=1
        transforms = Compose(
            [SpatialCrop(roi_center=center_of_mass(mask[0]),roi_size=patch_size),
            SpatialPad(spatial_size=patch_size)
            ]
        )
        img = transforms(image[0])
        return img[None,...]

    def validation(self, epoch):
        global n_iter
        self.model.eval()
        self.evaluator_seg_ctl.reset()
        self.evaluator_seg_jdm.reset()
        self.evaluator_cls.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        num_img_ts = len(self.val_loader)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image = sample['img'].to(self.device)
            target = sample['seg'].to(self.device)
            target_jdm = sample['jdm'].to(self.device)
            target_jdm_masked = sample['masked'].to(self.device)
            target_cls = sample['label'].to(self.device)
            target_cls_weight = sample['weight'].to(self.device)
            with torch.no_grad():
                output,output_jdm = sliding_window_inference(image, self.patch_size, self.args.solver.sw_batch_size, self.model,flag=True)
                # # Add batch sample into evaluator
                # During the verification period, the ground truth is used to select the patch, 
                # and the verification process is guided by a stable verification curve
                patch_image = self.get_patch_img(image, target, self.patch_size)
                patch_output,patch_output_jdm, output_cls = self.model(patch_image)

                self.evaluator_seg_ctl.add_batch(output,target)
                self.evaluator_seg_jdm.add_batch(output_jdm*target_jdm_masked,target_jdm)
                self.evaluator_cls.add_batch(output_cls, target_cls)

                loss_seg_ctl = self.criterion_seg_ctl(output, target)
                loss_seg_jdm = self.criterion_seg_jdm(output_jdm*target_jdm_masked, target_jdm)
                loss_cls = self.criterion_cls(output_cls, target_cls,target_cls_weight)
                loss = self.criterion(loss_seg_ctl,loss_seg_jdm,loss_cls)
                
                test_loss = loss.detach().item()+test_loss
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
       
       
        imgs_show = torchvision.utils.make_grid(image.as_tensor()[0,...].permute(3,0,1,2),normalize=True)
        masks_show = torchvision.utils.make_grid(target.as_tensor()[0,...].permute(3,0,1,2).float(),normalize=True)
        pred_show = torchvision.utils.make_grid(torch.argmax(output.as_tensor(),dim=1)[0,None,...].permute(3,0,1,2).float(),normalize=True)
        
        self.writer.add_image('mask/test',masks_show,epoch,dataformats='CHW')
        self.writer.add_image('mask_pred/test',pred_show,epoch,dataformats='CHW')
        self.writer.add_image('Img/test',imgs_show,epoch,dataformats='CHW')

        # Fast test during the training
        dice_ctl = self.evaluator_seg_ctl.Dice().cpu().item()
        dice_jdm = self.evaluator_seg_jdm.Dice().cpu().item()
        hd95_ctl = self.evaluator_seg_ctl.HD95().cpu().item()
        hd95_jdm = self.evaluator_seg_jdm.HD95().cpu().item()
        f1 = self.evaluator_cls.F1()
        acc = self.evaluator_cls.ACC()
        auc = self.evaluator_cls.AUC()
        sens = self.evaluator_cls.Recall(pos_label=1)
        spec = self.evaluator_cls.Recall(pos_label=0)
        cm = self.evaluator_cls.Confusion_matrix()
        report = self.evaluator_cls.Report()
        self.evaluator_seg_ctl.reset()
        self.evaluator_seg_jdm.reset()
        self.evaluator_cls.reset()

        self.writer.add_scalar('dice_ctl', dice_ctl, epoch)
        self.writer.add_scalar('dice_jdm', dice_jdm, epoch)
        self.writer.add_scalar('validation_loss', test_loss / i * self.args.solver.batch_size.test + image.data.shape[0], epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.solver.batch_size.test + image.data.shape[0]))
        print(f"dice_ctl: {dice_ctl:.4f}  hd95_ctl: {hd95_ctl:.2f}")
        print(f"dice_jdm: {dice_jdm:.4f} hd95_jdm: {hd95_jdm:.2f}")
        print(f"f1: {f1:.4f}  acc: {acc:.4f}  auc: {auc:.4f} sens: {sens:.4f}  spec: {spec:.4f}")
        print(cm)
        print(report)
        print("loss w:",self.criterion.get_params())
        print('Loss: %.3f' % (test_loss/num_img_ts))

        new_pred = auc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            

def main():
    parser = argparse.ArgumentParser(description="EBCV Classifcation Training")
    parser.add_argument('--configfile', type=str, default='configs/Config.yaml',  
                        help='config file path')

    args = parser.parse_args()
    global n_iter
    n_iter = 0
    print(args)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args.configfile)
    print('Starting Epoch:', trainer.args.solver.epoch_start)
    print('Total Epoches:', trainer.args.solver.epoch_max)
    for epoch in range(trainer.args.solver.epoch_start, trainer.args.solver.epoch_max):
        trainer.training(epoch)
        if epoch % trainer.args.solver.epoch_save == (trainer.args.solver.epoch_save - 1):
            trainer.validation(epoch)
    trainer.writer.close()

if __name__ == "__main__":
    log_path = './log'
    os.makedirs(log_path, exist_ok=True)
    make_print_to_file(log_path)
    main()
