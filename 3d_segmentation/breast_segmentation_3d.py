'''
  @ Date: 2022/6/20 20:43
  @ Author: Zhao YaChen
'''
# Setup imports
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset
from monai.losses import DiceLoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandFlipd,
    Spacingd,
    EnsureTyped,
    EnsureType, AddChanneld, ScaleIntensityd, ConcatItemsd,
)
import torch
from monai.utils import first
from networks.net_factory_3d import net_factory_3d

'''
3D segmentation  based on UNetModel
'''

# data_types = ['_ph1_voi_192x192x48.nii', '_ph3_voi_192x192x48.nii', '_ph5_voi_192x192x48.nii',
# '_seg_voi_192x192x48.nii']
data_types = ['_ph1_voi_128x128x48.nii', '_ph3_voi_128x128x48.nii', '_ph5_voi_128x128x48.nii',
              '_t2_sitk_voi_128x128x48.nii', '_dwi_sitk_voi_128x128x48.nii', '_seg_voi_128x128x48.nii']

data_types_name = ['dceph1', 'dceph3', 'dceph5', 't2', 'dwi', 'label']


def get_file_list():
    # Create MONAI dataset
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
        if name_mapping_df['labelSlicesum'][idx] == 9 or name_mapping_df['labelSlicesum'][idx] == 10 \
                or name_mapping_df['labelSlicesum'][idx] == 11 or name_mapping_df['labelSlicesum'][idx] == 12 \
                or name_mapping_df['labelSlicesum'][idx] == 13 or name_mapping_df['labelSlicesum'][idx] == 14 \
                or name_mapping_df['labelSlicesum'][idx] == 15:
            val_files.append(file)
        else:
            train_files.append(file)

    # train_files = files[:200]  # [0:215]   000-214
    # val_files = files[201:240]  # [215:265] 215-264
    # train_files.extend(files[240:])
    return train_files, val_files


def main():
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if pin_memory else "cpu")
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()
    # get files
    train_files, val_files = get_file_list()
    # define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["dceph3", "label"]),
            AddChanneld(keys=["dceph3", "label"]),
            # Spacingd(
            #     keys=["dceph3", "label"],
            #     pixdim=(0.5, 0.5, 1.5),
            #     mode=("bilinear", "nearest"),
            # ),
            # RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64], random_size=False),
            RandFlipd(keys=["dceph3", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["dceph3", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["dceph3", "label"], prob=0.5, spatial_axis=2),
            ScaleIntensityd(keys=["dceph3", "label"]),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # ConcatItemsd(keys=["dceph3", "t2", "dwi"], name="inputs"),
            EnsureTyped(keys=["dceph3", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["dceph3", "label"]),
            AddChanneld(keys=["dceph3", "label"]),
            # Spacingd(
            #     keys=["dceph3", "label"],
            #     pixdim=(0.5, 0.5, 1.5),
            #     mode=("bilinear", "nearest"),
            # ),
            ScaleIntensityd(keys=["dceph3", "label"]),
            # ConcatItemsd(keys=["dceph3", "t2", "dwi"], name="inputs"),
            EnsureTyped(keys=["dceph3", "label"]),
        ]
    )
    # check transforms in data loader
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["dceph3"][0][0], check_data["label"][0][0])
    print(f"dceph3 shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("dceph3")
    plt.imshow(image[:, :, 24], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 24])
    plt.show()

    # # Define dataset, data loader
    # check_ds = Dataset(data=train_files, transform=train_transforms)
    # check_loader = DataLoader(check_ds, batch_size=1, num_workers=4, pin_memory=pin_memory)
    # check_data = first(check_loader)
    # print(check_data["dceph3"].shape, check_data["inputs"].shape, check_data["label"].shape)
    # IMAGE_WIDTH = 128
    # # show check_data dceph1(1, 1, 128, 128, 48) + dceph3 + dceph5 + inputs(1, 3, 128, 128, 48)
    # sample_dce_image = torch.unsqueeze(torch.stack((check_data["dceph3"][0, 0, :, :, 0:41:10],
    #                                                 check_data["t2"][0, 0, :, :, 0:41:10],
    #                                                 check_data["dwi"][0, 0, :, :, 0:41:10]))
    #                                    .permute(0, 3, 1, 2).reshape(15, IMAGE_WIDTH, IMAGE_WIDTH), dim=1).repeat(1, 3,
    #                                                                                                              1, 1)
    # # (3, 128, 128, 5) -> (3, 5, 128, 128) -> (15, 128, 128) -> (15, 3, 128, 128)
    # grid_dce_image = make_grid(sample_dce_image, nrow=5, padding=0, normalize=False)  # nrow一行放5个
    # plt.imshow(np.transpose(grid_dce_image.numpy(), (1, 2, 0)))
    # plt.show()
    #
    # sample_inputs_image = torch.unsqueeze(check_data["inputs"][0, 0:3, :, :, 0:41:10].permute(0, 3, 1, 2)
    #                                       .reshape(15, IMAGE_WIDTH, IMAGE_WIDTH), dim=1).repeat(1, 3, 1, 1)
    # # (3, 128, 128, 5) -> (3, 5, 128, 128) -> (15, 128, 128) -> (15, 3, 128, 128)
    # grid_inputs_image = make_grid(sample_inputs_image, nrow=5, padding=0, normalize=False)  # nrow一行放5个
    # plt.imshow(np.transpose(grid_inputs_image.numpy(), (1, 2, 0)))
    # plt.show()

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=2, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # create model, loss, optimizer
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=2,
    #     channels=(16, 32, 64, 128, 256),
    #     dropout=0.1,
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    #     norm=Norm.BATCH,
    # ).to(device)
    model = net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2)
    loss_function = DiceFocalLoss(gamma=1.0, to_onehot_y=True, sigmoid=True)
    # loss_function = FocalLoss(to_onehot_y=False, gamma=1, reduction="mean", weight=1.0)
    # loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    max_epochs = 300
    # 学习率衰减策略之余弦退火学习率(Cosine Annealing LR)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # define inference method
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(128, 128, 64),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        return _compute(input)

    # Execute a typical PyTorch training process
    val_interval = 10
    iter_image_interval = 100
    best_metric = -1
    best_metric_epoch = -1
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    model_filename = "best_breast_unetmodel.pth"
    val_epoch = 0

    # load old model
    # pretrain_path = "D:/Desktop/MONAI/3d_segmentation/best_breast_unetmodel_epoch100_lr0.001_Adam.pth"
    # if not os.path.exists(pretrain_path):
    #     raise IOError(f"Pretrained model '{pretrain_path}' does not exist")
    # model.load_state_dict(torch.load(pretrain_path))

    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["dceph3"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_soft = torch.softmax(outputs, dim=1)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            iter_num = epoch_len * epoch + step
            writer.add_scalar("train/Step_train_loss", loss.item(), iter_num)

            if iter_num % iter_image_interval == 0:
                image = inputs[0, 0:1, :, :, 20:41:5].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:41:5].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, epoch_len * iter_num)

                image = labels[0, :, :, :, 20:41:5].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, epoch_len * iter_num)
        # lr_scheduler.step()
        # compute epoch loss
        epoch_loss /= step
        print(f"epoch {epoch + 1} train average loss: {epoch_loss:.4f}")
        writer.add_scalar("train/Train_loss", epoch_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                iteration = 0
                for val_data in val_loader:
                    iteration += 1
                    val_inputs, val_labels = (
                        val_data["dceph3"].to(device),
                        val_data["label"].to(device)
                    )
                    # val_outputs = inference(val_inputs)
                    roi_size = (128, 128, 48)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    # val_outputs = model(val_inputs)
                    val_loss = loss_function(val_outputs, val_labels)
                    val_epoch_loss += val_loss.item()
                    # compute metric for current iteration
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    val_epoch_len = len(val_ds) // val_loader.batch_size
                    writer.add_scalar("val/Step_val_loss", val_loss.item(), val_epoch_len * val_epoch + iteration)
                # compute epoch loss
                val_epoch_loss /= iteration
                writer.add_scalar("val/Val_loss", val_epoch_loss, epoch + 1)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                writer.add_scalar("val/Val_metric", metric, epoch + 1)
                # reset the status for next validation round
                dice_metric.reset()
                val_epoch += 1
                print(f"epoch {epoch + 1} val average loss: {val_epoch_loss:.4f} val metric: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == '__main__':
    main()
