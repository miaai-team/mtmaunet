from cProfile import label
import random
from re import S
import torch 
# from torch.utils.data import Dataset 
import os.path as osp
import numpy as np
import copy


from monai.transforms import (
    AsChannelFirstd,
    ScaleIntensityd,
    AddChanneld,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandRotate90d,
    EnsureTyped,
    Compose,
    LoadImaged,
    SaveImaged,
    SaveImage,
    KeepLargestConnectedComponentd,
    AsDiscreted,
    Spacingd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandFlipd,
    SpatialPadd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    CenterSpatialCropd,
    RandSpatialCropd,
    RandWeightedCropd,
    CropForegroundd,
    Invertd,
    EnsureChannelFirstd,
    Activationsd,
    Orientationd,
    ToDeviced,
)

from monai.transforms import SpatialCrop,SpatialPad, Compose
from scipy.ndimage.measurements import center_of_mass

from monai.transforms.compose import MapTransform

from monai.data import CacheDataset, LMDBDataset

from monai.apps import DecathlonDataset

from scipy.ndimage import distance_transform_edt as distance
 

class myDatasetInfer(CacheDataset):
    def __init__(self, root, csv_path):
        self.root = root
        self.csv_path = csv_path
        self._set_files()
        self._set_transforms()
        CacheDataset.__init__(self,self.files,self.transforms,cache_rate=0)
        
    def _set_files(self):
        self.files = []
        with open(self.csv_path) as f:
            contents = f.readlines()
            for n in contents:
                image = n.strip().split(',')
                self.files.append({"img": osp.join(self.root,image)})


    def _set_transforms(self):
        self.transforms = Compose(
            [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            Spacingd(
                keys=["img"],
                pixdim=(0.3515625, 0.3515625, 2.0),
                mode=("bilinear"),
            ),
            NormalizeIntensityd(keys="img",nonzero=True, channel_wise=True),
            EnsureTyped(keys=["img"]),
            ])      

        self.post_transforms = Compose([
            Activationsd(keys="pred",softmax=True),
            Invertd(
                keys="pred",
                transform=self.transforms,
                orig_keys="img",
                meta_keys="pred_meta_dict",
                orig_meta_keys="img_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                # to_tensor=False,
                # device="cpu",
            ),
            AsDiscreted(keys="pred",argmax=True),
            KeepLargestConnectedComponentd(keys="pred"),
            SaveImaged(
                keys="pred",
                output_dir = "./pred",
                output_postfix= "seg",
                output_ext= ".nrrd",
                output_dtype = np.uint8,
                separate_folder=False,
                print_log=False,
            ),
            EnsureTyped(keys=["pred"]),
            ToDeviced(keys=["pred"],device='cpu'),
        ])




class myDataset(CacheDataset):
    def __init__(self, root, csv_path,no_channel,patch_size, num_classes=4, is_train=False):
        self.root = root
        self.csv_path = csv_path
        self.is_train = is_train
        self.no_channel = no_channel
        self.patch_size = patch_size
        self.num_classes = num_classes
        self._set_files()
        self._set_transforms()
        CacheDataset.__init__(self,self.files,self.transforms,cache_rate=0)
        
# class myDataset(LMDBDataset):
#     def __init__(self, root, csv_path,no_channel,patch_size, num_classes=4, is_train=False):
#         self.root = root
#         self.csv_path = csv_path
#         self.is_train = is_train
#         self.no_channel = no_channel
#         self.patch_size = patch_size
#         self.num_classes = num_classes
#         self._set_files()
#         self._set_transforms()
#         LMDBDataset.__init__(self,self.files,self.transforms,cache_dir='./cache_dir')
        
        
    def _set_files(self):
        self.files = []
        with open(self.csv_path) as f:
            contents = f.readlines()
            for n in contents:
                image,mask,label = n.strip().split(',')
                if int(label)==1 and self.is_train:
                    self.files.append({"img": osp.join(self.root,image), "seg": osp.join(self.root,mask), "label": int(label)})
                    self.files.append({"img": osp.join(self.root,image), "seg": osp.join(self.root,mask), "label": int(label)})
                else:
                    self.files.append({"img": osp.join(self.root,image), "seg": osp.join(self.root,mask), "label": int(label)})

    def _set_transforms(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                AddChanneld(keys=["img", "seg"]),
                Segtransform(keys=["img", "seg"]),
                Spacingd(
                    keys=["img", "seg"],
                    pixdim=(0.3515625, 0.3515625, 2.0),
                    mode=("bilinear", "nearest"),
                   
                ),
                NormalizeIntensityd(keys="img",nonzero=True, channel_wise=True),
                CropMy(keys=["img", "seg"]),
                Rrea(keys=["seg"]),
                SpatialPadd(keys=["img", "seg"], spatial_size=self.patch_size),
                RandSpatialCropd(keys=["img", "seg"], roi_size=self.patch_size,random_center=True,random_size=False),
                PatchLabeltransform(keys=["img", "seg", "area"]),
                RandZoomd(
                keys=["img", "seg"],
                min_zoom=0.8,
                max_zoom=1.2,
                mode=("trilinear", "nearest"),
                align_corners=(True, None),
                prob=0.15,
                ),
                RandGaussianNoised(keys=["img"], std=0.01, prob=0.15),
                RandGaussianSmoothd(
                    keys=["img"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["img"], factors=0.3, prob=0.15),
                RandFlipd(["img", "seg"], spatial_axis=[0], prob=0.5),
                RandFlipd(["img", "seg"], spatial_axis=[1], prob=0.5),
                RandFlipd(["img", "seg"], spatial_axis=[2], prob=0.5),
                RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 1]),
                SegSplit(keys="seg"),
                Masked(keys="jdm"),
                EnsureTyped(keys=["img", "seg", "jdm",'masked']),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                AddChanneld(keys=["img", "seg"]),
                Segtransform(keys=["img", "seg"]),
                Spacingd(
                    keys=["img", "seg"],
                    pixdim=(0.3515625, 0.3515625, 2.0),
                    mode=("bilinear", "nearest"),
                   
                ),
                NormalizeIntensityd(keys="img",nonzero=True, channel_wise=True),
                SegSplit(keys="seg"),
                Masked(keys="jdm"),
                EnsureTyped(keys=["img", "seg", "jdm",'masked']),
            ]
        )
        self.transforms = train_transforms if self.is_train else val_transforms
                

class Segtransform(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        # 1 颈动脉（左） 
        # 2 颈动脉（右）
        # 3 垂体瘤
        d = dict(data)
        seg = d["seg"]
        assert seg.ndim == 4
        seg[seg==2] = 1
        seg[seg>2] = 2
        d["seg"] = seg
        d["weight"]= 1
        return d

class SegSplit(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        # 1 颈动脉 
        # 2 垂体瘤
        d = dict(data)
        seg = d["seg"]
        assert seg.ndim == 4
        seg_ctl =  copy.deepcopy(seg)
        seg_jdm =  copy.deepcopy(seg)

        seg_ctl[seg_ctl<2]=0
        seg_ctl[seg_ctl>0]=1

        seg_jdm[seg_jdm>1]=0

        assert len(np.unique(seg_ctl))<=2
        assert len(np.unique(seg_jdm))<=2

        d["seg"] = seg_ctl
        d["jdm"] = seg_jdm
        return d

class Rrea(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        seg = d["seg"]
        temp_seg = copy.deepcopy(seg)
        temp_seg[temp_seg>0]=1 
        area = np.sum(temp_seg,axis=(1,2,3))[0]
        d["area"] = area
        return d

class PatchLabeltransform(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        # load data
        d = dict(data)
        seg = d["seg"]
        area = d["area"]
        assert seg.ndim==4
        temp_seg = copy.deepcopy(seg)
        temp_seg[temp_seg>0]=1
        iou = np.sum(temp_seg,axis=(1,2,3))[0]/area
        d["weight"]= iou
        return d

class CropMy(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        # load data
        d = dict(data)
        num = np.random.random(1)
        # print(num)
        if num<=0.2:
            img = d["img"]
            seg = d["seg"]
            mask = copy.deepcopy(seg)
            mask = np.sum(mask,axis=1)
            transforms = SpatialCrop(roi_center=center_of_mass(mask),roi_size=[384,384,32])
            d["img"] = transforms(img)
            d["seg"] = transforms(seg)
        if 0.2<num<=0.5:
            img = d["img"]
            seg = d["seg"]
            mask = copy.deepcopy(seg)
            mask = np.sum(mask,axis=1)
            transforms = SpatialCrop(roi_center=center_of_mass(mask),roi_size=[448,448,32])
            d["img"] = transforms(img)
            d["seg"] = transforms(seg)
        return d

class Masked(MapTransform):
    def __init__(
        self,
        keys
    ) -> None:
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        # load data
        d = dict(data)
        seg = d['jdm']
        if seg.max()==0:
            d['masked']=np.zeros_like(d['jdm'])
            return d
        seg = np.sum(seg,axis=(0,1,2))
        seg[seg>=1]=1
        seg[seg<1]=0
        d['masked']=seg[None,...]*np.ones_like(d['jdm'])
        return d


