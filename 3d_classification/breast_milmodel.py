'''
  @ Date: 2022/4/24 15:51
  @ Author: Zhao YaChen
'''
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchsummary import summary

import monai.networks.nets
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.transforms import LoadImaged, AddChanneld, ConcatItemsd, RandRotate90d, ScaleIntensityd, ToTensord, \
    EnsureType, Activations, AsDiscrete, Orientationd, Spacingd, ScaleIntensityRanged, NormalizeIntensityd, \
    HistogramNormalized, RandBiasFieldd, RandAffined, RandGaussianNoised, RandFlipd
from monai.networks.nets import mymilmodel
from monai.transforms import Compose
from monai.losses import FocalLoss


'''
3D classification  based on MILModel
Here, the task is given to classify MR images into pCR/non-pCR.
'''

data_types = ['_ph1_voi_192x192x48.nii', '_ph3_voi_192x192x48.nii', '_ph5_voi_192x192x48.nii']
data_types_name = ['dceph1', 'dceph3', 'dceph5']
def main():
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if pin_memory else "cpu")
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Create MONAI dataset
    default_prefix = 'D:/Desktop/BREAST/BREAST/'
    dce_train_data = default_prefix + 'breast-dataset-training-validation/Breast_TrainingData'
    name_mapping_path = default_prefix + 'breast-dataset-training-validation/Breast_meta_data/breast_name_mapping.csv'

    # 使用df读取Breast_MR_list.xlsx文件
    pCR_info_df = pd.read_csv(
        'D:/Desktop/BREAST/BREAST/breast-dataset-training-validation/Breast_meta_data/Breast_MR_list_ori.csv')
    name_mapping_df = pd.read_csv(name_mapping_path)
    name_mapping_df.rename({'Number': 'ID'}, axis=1, inplace=True)
    df = pCR_info_df.merge(name_mapping_df, on="ID", how="right")

    name_mapping_path = name_mapping_df['Breast_subject_ID'].tolist()

    files = []
    for id_ in name_mapping_path:
        file = {}
        for data_type, data_type_name in zip(data_types, data_types_name):
            file[data_type_name] = os.path.join(dce_train_data, id_, id_ + data_type)
        index = df['Breast_subject_ID'][df['Breast_subject_ID'].values == id_].index
        pCR_label = df['病理完全缓解'][index.values[0]]  # 0/1
        file["label"] = pCR_label
        files.append(file)

    train_files = files[:215]  # [0:215]   000-214
    val_files = files[215:265]    # [215:265] 215-264
    train_files.extend(files[265:])

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["dceph1", "dceph3", "dceph5"]),
            AddChanneld(keys=["dceph1", "dceph3", "dceph5"]),
            # Resized(keys=["dceph1", "dceph3", "dceph5"], spatial_size=(32, 32, 32)),
            # Orientationd(keys=["dceph1", "dceph3", "dceph5"], axcodes="RAS"),  # 方向变换
            # Spacingd( keys=["image", "label"],pixdim=(0.5, 0.5, 1.5),mode=("bilinear", "nearest"),),
            # Spacingd(keys=["dceph1", "dceph3", "dceph5"], pixdim=(0.5, 0.5, 1.5)),  # 体素间距变换
            # RandBiasFieldd(keys=["dceph1", "dceph3", "dceph5"]),
            # ScaleIntensityRanged(keys=["dceph1", "dceph3", "dceph5"], a_min=0, a_max=3500.0, b_min=0.0, b_max=1.0, clip=True),  # 强度变换
            ScaleIntensityd(keys=["dceph1", "dceph3", "dceph5"]),
            # NormalizeIntensityd(keys=["dceph1", "dceph3", "dceph5"], nonzero=True, channel_wise=True),
            # HistogramNormalized(keys=["dceph1", "dceph3", "dceph5"]),
            # RandAffined(  # 放射变换
            #     keys=["dceph1", "dceph3", "dceph5"],
            #     prob=0.15,
            #     rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
            #     scale_range=(0.1, 0.1, None),
            #     mode=("bilinear", "nearest"),
            #     as_tensor_output=False,
            # ),
            RandGaussianNoised(keys=["dceph1", "dceph3", "dceph5"], prob=0.15, std=0.01),  # 添加高斯噪声
            RandFlipd(keys=["dceph1", "dceph3", "dceph5"], spatial_axis=0, prob=0.5),  # 随机翻转
            RandFlipd(keys=["dceph1", "dceph3", "dceph5"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["dceph1", "dceph3", "dceph5"], spatial_axis=2, prob=0.5),
            # ConcatItemsd(keys=["dceph1", "dceph3", "dceph5"], name="inputs"),
            ConcatItemsd(keys=["dceph3"], name="inputs"),

            # ScaleIntensityd(keys=["inputs"]),
            # RandRotate90d(keys=["inputs"], prob=0.8, spatial_axes=[0, 1]),
            ToTensord(keys=["inputs"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["dceph1", "dceph3", "dceph5"]),
            AddChanneld(keys=["dceph1", "dceph3", "dceph5"]),
            # Resized(keys=["dceph1", "dceph3", "dceph5"], spatial_size=(32, 32, 32)),
            ScaleIntensityd(keys=["dceph1", "dceph3", "dceph5"]),
            # ConcatItemsd(keys=["dceph1", "dceph3", "dceph5"], name="inputs"),
            ConcatItemsd(keys=["dceph3"], name="inputs"),
            # ScaleIntensityd(keys=["inputs"]),
            ToTensord(keys=["inputs"]),
        ]
    )
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1, num_workers=4, pin_memory=pin_memory)
    # check data
    def lala(iterable, default=None):
        for i, index in zip(check_loader, range(10)):
            if index == 8:
                return i
        return default
    check_data = lala(check_loader)
    # check_data = monai.utils.misc.first(check_loader)
    print(check_data["dceph1"].shape, check_data["inputs"].shape, check_data["label"])
    IMAGE_WIDTH = 192
    # show check_data dceph1(1, 1, 128, 128, 48) + dceph3 + dceph5 + inputs(1, 3, 128, 128, 48)
    sample_dce_image = torch.unsqueeze(torch.stack((check_data["dceph1"][0, 0, :, :, 0:41:10],
                                 check_data["dceph3"][0, 0, :, :, 0:41:10],
                                 check_data["dceph5"][0, 0, :, :, 0:41:10]))
                                       .permute(0, 3, 1, 2).reshape(15, IMAGE_WIDTH, IMAGE_WIDTH), dim=1).repeat(1, 3, 1, 1)
    # (3, 128, 128, 5) -> (3, 5, 128, 128) -> (15, 128, 128) -> (15, 3, 128, 128)
    grid_dce_image = make_grid(sample_dce_image, nrow=5, padding=0, normalize=False)  # nrow一行放5个
    plt.imshow(np.transpose(grid_dce_image.numpy(), (1, 2, 0)))
    plt.show()
    sample_inputs_image = torch.unsqueeze(check_data["inputs"][0, 0:3, :, :, 0:41:10].permute(0, 3, 1, 2)
                                          .reshape(15, IMAGE_WIDTH, IMAGE_WIDTH), dim=1).repeat(1, 3, 1, 1)
    # (3, 128, 128, 5) -> (3, 5, 128, 128) -> (15, 128, 128) -> (15, 3, 128, 128)
    grid_inputs_image = make_grid(sample_inputs_image, nrow=5, padding=0, normalize=False)  # nrow一行放5个
    plt.imshow(np.transpose(grid_inputs_image.numpy(), (1, 2, 0)))
    plt.show()


    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=3, shuffle=True, num_workers=4, pin_memory=pin_memory)

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=3, num_workers=4, pin_memory=pin_memory)

    # Create MILModel, CrossEntropyLoss and Adam optimizer
    # model = mymilmodel.MYMILModel(num_classes=2, pretrained=True, mil_mode="att").to(device)
    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    # model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, n_classes=2).to(device)
    model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, n_classes=2).to(device)
    # model = monai.networks.nets.resnet101(spatial_dims=3, n_input_channels=3, n_classes=2).to(device)
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=3, out_channels=2).to(device)
    summary(model, (1, 128, 128, 48))
    weight_CE = torch.FloatTensor([0.65, 0.35]).to(device)
    # loss_function = torch.nn.CrossEntropyLoss(weight=weight_CE)
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = FocalLoss(to_onehot_y=False, gamma=1, reduction="mean", weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    auc_metric = ROCAUCMetric()

    # Start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    max_epochs = 260
    epoch_loss_values = []
    metric_values = []
    model_filename = "best_breast_resnetmodel_adam_bs3_1e-5_CE_192x192x48_GCY_weights.pth"

    # checkpoint_path = os.path.join(os.getcwd(), "best_breast_densenetmodel_0426.pth")
    # if not os.path.exists(checkpoint_path):
    #     raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")
    # model.load_state_dict(torch.load(checkpoint_path))

    pretrain_path = "D:/Downloads/MedicalNet_pytorch_files/pretrain/resnet_50.pth"
    if not os.path.exists(pretrain_path):
        raise IOError(f"Pretrained model '{pretrain_path}' does not exist")
    # model.load_state_dict(torch.load(pretrain_path))
    weights_dict = torch.load(pretrain_path)
    weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}
    model_dict = model.state_dict()
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        step = 0
        train_y_pred = torch.tensor([], dtype=torch.float32, device=device)
        train_y = torch.tensor([], dtype=torch.long, device=device)
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["inputs"].to(device), batch_data["label"].to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=2).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            train_y_pred = torch.cat([train_y_pred, outputs], dim=0)
            train_y = torch.cat([train_y, labels], dim=0)
            # compute loss
            loss = loss_function(outputs, labels)
            # loss = loss_function(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("step_train_loss", loss.item(), epoch_len * epoch + step)
        # compute accuracy
        train_acc_value = torch.eq(train_y_pred.argmax(dim=1), train_y)
        train_acc_metric = train_acc_value.sum().item() / len(train_acc_value)
        # compute epoch loss
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} train accuracy: {train_acc_metric:.4f}")
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        writer.add_scalar("train_accuracy", train_acc_metric, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                iteration = 0
                for val_data in val_loader:
                    iteration += 1
                    val_images, val_labels = val_data["inputs"].to(device), val_data["label"].to(device)
                    val_labels_onehot = torch.nn.functional.one_hot(val_labels, num_classes=2).float()
                    val_outputs = model(val_images)
                    y_pred = torch.cat([y_pred, val_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                    val_loss = loss_function(val_outputs, val_labels)
                    # val_loss = loss_function(val_outputs, val_labels_onehot)
                    val_epoch_loss += val_loss.item()
                    val_epoch_len = len(val_ds) // val_loader.batch_size
                    writer.add_scalar("step_val_loss", val_loss.item(), val_epoch_len * epoch + iteration)

                # compute epoch loss
                val_epoch_loss /= iteration
                # compute accuracy
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                # decollate prediction and label and execute post processing
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                y_onehot = [post_label(i) for i in decollate_batch(y)]
                # compute AUC
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                metric_values.append(auc_result)
                del y_pred_act, y_onehot

                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), model_filename)
                    print("saved new best metric model")
                print(
                    "current epoch: {} average loss: {:.4f} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, val_epoch_loss, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_loss", val_epoch_loss, epoch + 1)
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
                writer.add_scalar("val_AUC", auc_result, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == '__main__':
    main()