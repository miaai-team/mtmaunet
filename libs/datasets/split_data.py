import os
from os.path import join, isdir, isfile
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

import yaml
from addict import Dict

def write_image_label_txt(img_label, txt_path):
    with open(txt_path, 'w') as f:
        for img_path, label in img_label:
            f.write(img_path+',')
            f.write(str(label)+'\n')

def split_dataset_with_cv(config_path):
    
    config = Dict(yaml.load(open(config_path,'r'), Loader=yaml.FullLoader))
    
    k = config.dataset.cv.num
    root = config.dataset.root
    csv_path = config.dataset.csv_path
    cv_dir = config.dataset.cv.dir_name
    shuffle = config.dataset.cv.shuffle 
    random_state = config.dataset.cv.random_state
    fold_dir = join(root,cv_dir)
    
    assert k>1, 'k must > 1'
    assert isdir(root), 'root is not find'
    assert isfile(csv_path), 'csv is not find'
    
    if isfile(join(fold_dir,f'fold_0',f'train.csv')):
        print('File already exists, skip. If you need to re-split please delete the CV folder\n')
        return
    
    os.makedirs(fold_dir,exist_ok=True)
    data = np.array(pd.read_csv(csv_path,header=None))
    skf = KFold(n_splits=k,shuffle=shuffle,random_state=random_state)
    i = 0
    for i,(train_index, test_index) in enumerate(skf.split(data)):
        os.makedirs(join(fold_dir,f'fold_{i}'),exist_ok=True)
        write_image_label_txt(data[train_index].tolist(), txt_path=join(fold_dir,f'fold_{i}',f'train.csv'))
        write_image_label_txt(data[test_index].tolist(), txt_path=join(fold_dir,f'fold_{i}',f'val.csv'))
        
    