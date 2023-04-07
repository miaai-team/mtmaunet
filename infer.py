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
from libs.datasets.base import myDatasetInfer
from libs.datasets.split_data import split_dataset_with_cv
from libs.utils import saver, metric, LR_Scheduler, make_print_to_file
from tensorboardX import SummaryWriter
import datetime 

from monai.inferers import sliding_window_inference
from monai.data import create_test_image_3d, list_data_collate, decollate_batch

from libs.loss import get_lossfunction, AutomaticWeightedLoss, FocalLoss_cls,SampleWeightedCELoss, DiceCELoss, DiceLoss
from monai.transforms import SpatialCrop,SpatialPad, Compose, SaveImaged
from scipy.ndimage.measurements import center_of_mass

from monai.handlers.utils import from_engine

from sklearn.metrics import classification_report, confusion_matrix

import torchvision.utils as vutils
import torchvision
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1024*8, rlimit[1]))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Inferencer(object):
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
        test_csv_path = self.args.dataset.test

        ## define dataloader of train and validation
        test_dataset = myDatasetInfer(
            root = self.args.dataset.root,
            csv_path = test_csv_path,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=self.args.dataloader.num_workers,
            shuffle=False,
            collate_fn=list_data_collate,
        )

        self.post_transforms = test_dataset.post_transforms
                     
        # Define network
        network_cls = get_network(config)
        network_param = {k: v for k, v in config['network'][config["network"]["type"]].items() if k != 'name'}
        self.model = network_cls(**network_param)
        
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


    def inference(self):
        global n_iter
        self.model.eval()
        for i, sample in enumerate(self.test_loader):
            image = sample['img'].to(self.device)
            file_path = sample['img_meta_dict']['filename_or_obj'][0]
            assert len(image)==1, 'infer batch must 1'
            
            with torch.no_grad():
                output,output_jdm = sliding_window_inference(image, self.patch_size, self.args.solver.sw_batch_size, self.model,overlap=0.5, flag=True)
                # # Add batch sample into evaluator
                patch_image = self.get_patch_img(image, output, self.patch_size)
                patch_output,patch_output_jdm, output_cls = self.model(patch_image)
                sample["pred"] = output
                test_data = [self.post_transforms(i) for i in decollate_batch(sample)]
                print(file_path,'cls: ',torch.argmax(output_cls,dim=1).item())


def main():
    parser = argparse.ArgumentParser(description="MTMAUnet")
    parser.add_argument('--configfile', type=str, default='configs/Config.yaml',  
                        help='config file path')

    args = parser.parse_args()
    global n_iter
    n_iter = 0
    print(args)
    torch.backends.cudnn.benchmark = True
    infer = Inferencer(args.configfile)
    infer.inference()

if __name__ == "__main__":
    log_path = './log_infer'
    filename = None
    os.makedirs(log_path, exist_ok=True)
    make_print_to_file(log_path,fileName=filename)
    main()
