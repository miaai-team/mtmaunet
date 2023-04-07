import argparse
from fileinput import filename
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
from libs.loss import get_lossfunction
from libs.datasets.base import myDataset
from libs.datasets.split_data import split_dataset_with_cv
from libs.utils import saver, metric, LR_Scheduler, make_print_to_file
from tensorboardX import SummaryWriter
import datetime 

import monai
from monai.inferers import sliding_window_inference
from monai.data import create_test_image_3d, list_data_collate, decollate_batch

from libs.loss import get_lossfunction, AutomaticWeightedLoss, FocalLoss_cls,SampleWeightedCELoss, DiceCELoss, DiceLoss
from monai.transforms import SpatialCrop,SpatialPad, Compose
from scipy.ndimage.measurements import center_of_mass


from sklearn.metrics import classification_report, confusion_matrix

import torchvision.utils as vutils
import torchvision
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024*8, rlimit[1]))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class Validationer(object):
    def __init__(self, config_path):
        config = Dict(yaml.load(open(config_path,'r'), Loader=yaml.FullLoader))
        self.args = config 
        
        ## Define accelerator
        accelerator_param = {k: v for k, v in config['exp']['accelerator'].items()}
        self.accelerator = Accelerator(**accelerator_param)
        self.device = self.accelerator.device
        ## Define Saver
        self.saver = saver.Saver(self.args, config_path)
        
        ## Get confige
        self.dim = self.args.dataset.dim
        self.channel = self.args.dataset.channel
        self.n_classes = self.args.dataset.n_classes
        self.patch_size = self.args.dataset.patch_size
        
        ## Get Dataset arg
        assert self.args.dataset.cv.fold < self.args.dataset.cv.num, 'fold too big'
        
        fold_i_path = os.path.join(self.args.dataset.root,self.args.dataset.cv.dir_name,f'fold_{self.args.dataset.cv.fold}')
        val_csv_path = os.path.join(fold_i_path,self.args.dataset.split.val)

        ## Define Evaluator
        self.evaluator_seg_ctl = metric.Evaluator_Seg(self.n_classes,include_background=False, reduction="mean")
        self.evaluator_seg_jdm = metric.Evaluator_Seg(self.n_classes,include_background=False, reduction="mean")
        self.evaluator_cls = metric.Evaluator_Cls(2)

        ## define dataloader of train and validation
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
        self.writer = SummaryWriter(log_dir='runs_val/{}/fold_{}/{}/{}'.format(self.args.exp.id, self.args.dataset.cv.fold, self.saver.id,datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

        if not os.path.isfile(self.args.val_model):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.args.val_model))
        checkpoint = torch.load(self.args.val_model)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) (best_pred)"
                .format(self.args.val_model, checkpoint['epoch'], checkpoint['best_pred']))

        #  Device free
        self.model= self.accelerator.prepare(self.model)


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

    @staticmethod
    def totensor(tensor):
        if isinstance(tensor,monai.data.meta_tensor.MetaTensor):
            return tensor.as_tensor()
        else:
            return tensor
        
    def validation(self):
        global n_iter
        self.model.eval()
        self.evaluator_seg_ctl.reset()
        self.evaluator_seg_jdm.reset()
        self.evaluator_cls.reset()

        tbar = tqdm(self.val_loader, desc='\r')
        num_img_ts = len(self.val_loader)

        for i, sample in enumerate(tbar):
            image = sample['img'].to(self.device)
            target = sample['seg'].to(self.device)
            target_cls = sample['label'].to(self.device)

            with torch.no_grad():
                output,output_jdm = sliding_window_inference(image, self.patch_size, self.args.solver.sw_batch_size, self.model,flag=True)
                # # Add batch sample into evaluator
                patch_image = self.get_patch_img(image, output, self.patch_size)
                patch_output,patch_output_jdm, output_cls = self.model(patch_image)

                self.evaluator_seg_ctl.add_batch(output.detach(), target.detach())
                self.evaluator_cls.add_batch(output_cls.detach(), target_cls.detach())
                
            imgs_show = torchvision.utils.make_grid(self.totensor(image)[0,...].permute(3,0,1,2),normalize=True)
            masks_show = torchvision.utils.make_grid(self.totensor(target)[0,...].permute(3,0,1,2).float(),normalize=True)
            pred_show = torchvision.utils.make_grid(self.totensor(torch.argmax(output,dim=1))[0,...].permute(3,0,1,2).float(),normalize=True)

            
            self.writer.add_image('mask/test',masks_show,i,dataformats='CHW')
            self.writer.add_image('mask_pred/test',pred_show,i,dataformats='CHW')
            self.writer.add_image('Img/test',imgs_show,i,dataformats='CHW')

        # Fast test during the training
        dice_ctl = self.evaluator_seg_ctl.Dice().cpu().item()
        hd95_ctl = self.evaluator_seg_ctl.HD95().cpu().item()
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

        print('Validation:')
        print('numImages: %5d' % (i * self.args.solver.batch_size.test + image.data.shape[0]))
        print(f"dice_ctl: {dice_ctl:.4f}  hd95_ctl: {hd95_ctl:.2f}")
        print(f"f1: {f1:.4f}  acc: {acc:.4f}  auc: {auc:.4f} sens: {sens:.4f}  spec: {spec:.4f}")
        print(cm)
        print(report)
       

def main():
    parser = argparse.ArgumentParser(description="EBCV")
    parser.add_argument('--configfile', type=str, default='configs/Config.yaml', 
                        help='config file path')

    args = parser.parse_args()
    global n_iter
    n_iter = 0
    print(args)
    torch.backends.cudnn.benchmark = True
    trainer = Validationer(args.configfile)
    trainer.validation()
    trainer.writer.close()

if __name__ == "__main__":
    log_path = './log_val'
    filename = None
    os.makedirs(log_path, exist_ok=True)
    make_print_to_file(log_path,fileName=filename)
    main()
