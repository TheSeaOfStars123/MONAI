'''
  @ Date: 2022/10/17 15:35
  @ Author: Zhao YaChen
'''
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from skimage.util import montage

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, EnsureTyped, SpatialPad

# data_types = ['_ph1_voi_128x128x48.nii', '_ph3_voi_128x128x48.nii', '_ph5_voi_128x128x48.nii',
#               '_t2_sitk_voi_128x128x48.nii', '_dwi_sitk_voi_128x128x48.nii', '_seg_voi_128x128x48.nii']

data_types = ['_ph1_voi.nii', '_ph3_voi_128x128x48.nii', '_ph5_voi_128x128x48.nii',
              '_t2_sitk_voi_128x128x48.nii', '_dwi_sitk_voi_128x128x48.nii', '_seg_voi_128x128x48.nii']

data_types_name = ['dceph1', 'dceph3', 'dceph5', 't2', 'dwi', 'label']


def get_file_list():
    default_prefix = 'D:/Desktop/BREAST/BREAST/'
    dce_train_data = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
    name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_path = name_mapping_df['Breast_subject_ID'].tolist()

    val_files = []
    train_files = []
    for idx, id_ in enumerate(name_mapping_path):
        file = {}
        for data_type, data_type_name in zip(data_types, data_types_name):
            file[data_type_name] = os.path.join(dce_train_data, id_, id_ + data_type)

        train_files.append(file)
    return train_files, val_files


def save_as_montage():
    pin_memory = torch.cuda.is_available()
    train_files, val_files = get_file_list()
    train_transforms = Compose(
        [
            LoadImaged(keys=["dceph3", "t2", "dwi", "label"]),
            AddChanneld(keys=["dceph3", "t2", "dwi", "label"]),
            ScaleIntensityd(keys=["dceph3", "t2", "dwi", "label"]),
            EnsureTyped(keys=["dceph3", "t2", "dwi", "label"]),  # 确保输入数据为PyTorch Tensor或numpy数组
        ]
    )
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=pin_memory)
    for i, data in enumerate(train_loader):
        #  torch.Size([1, 1, 128, 128, 48])
        #  torch.Size([1, 1, 128, 128, 48])
        print(data["dceph3"].shape, data["label"].shape)
        dce_tensor = data['dceph3'].squeeze().cpu().detach().numpy()
        t2_tensor = data['t2'].squeeze().cpu().detach().numpy()
        dwi_tensor = data['dwi'].squeeze().cpu().detach().numpy()
        mask_tensor = data['label'].squeeze().cpu().detach().numpy()

        mask = np.rot90(montage(mask_tensor))
        fig_name = data['dceph1'][0].split("\\")[1]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 30))
        ax1.imshow(np.rot90(montage(dce_tensor)), cmap='bone')
        ax1.imshow(np.ma.masked_where(mask==False, mask), cmap='cool', alpha=0.6)
        ax1.set_title(fig_name+'_dceph3', fontsize=10)
        ax2.imshow(np.rot90(montage(t2_tensor)), cmap='bone')
        ax2.imshow(np.ma.masked_where(mask == False, mask), cmap='cool', alpha=0.6)
        ax2.set_title(fig_name + '_t2', fontsize=10)
        ax3.imshow(np.rot90(montage(dwi_tensor)), cmap='bone')
        ax3.imshow(np.ma.masked_where(mask == False, mask), cmap='cool', alpha=0.6)
        ax3.set_title(fig_name + '_dwi', fontsize=10)
        plt.savefig(fig_name + '.png')
        plt.show()

if __name__ == '__main__':
    save_as_montage()
