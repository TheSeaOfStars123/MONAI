'''
  @ Date: 2022/4/11 19:46
  @ Author: Zhao YaChen
'''
import logging
import os
import sys
import tempfile

import numpy as np
import monai
from monai.data import decollate_batch
from monai.apps import download_and_extract
from monai.config import print_config
from monai.transforms import Activations, AddChanneld, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd, EnsureTyped, EnsureType
from monai.metrics import ROCAUCMetric
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''
3D classification example based on DenseNet
This  tutorial shows an example of 3D classification task based on DenseNet and array format transforms.
Here, the task is given to classify MR images into male/female.
'''

def main():
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if pin_memory else "cpu")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    directory = "D:\Desktop\MONAI_DATA_DIRECTORY"
    # set this in your environment or previous cell to wherever IXI is downloaded and extracted
    # directory = os.environ.get("MONAI_DATA_DIRECTORY")

    if directory is None:
        resource = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
        md5 = "34901a0593b41dd19c1a1f746eac2d58"

        root_dir = tempfile.mkdtemp()

        dataset_dir = os.path.join(root_dir, "ixi")
        tarfile_name = f"{dataset_dir}.tar"

        download_and_extract(resource, tarfile_name, dataset_dir, md5)
    else:
        root_dir = directory

    print(root_dir)
    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    images = [
        os.sep.join([root_dir, "ixi", "IXI314-IOP-0889-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI249-Guys-1072-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI609-HH-2600-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI173-HH-1590-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI020-Guys-0700-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI342-Guys-0909-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI134-Guys-0780-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI577-HH-2661-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI066-Guys-0731-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI130-HH-1528-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI607-Guys-1097-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI175-HH-1570-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI385-HH-2078-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI344-Guys-0905-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI409-Guys-0960-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI584-Guys-1129-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI253-HH-1694-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI092-HH-1436-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI574-IOP-1156-T1.nii.gz"]),
        os.sep.join([root_dir, "ixi", "IXI585-Guys-1130-T1.nii.gz"]),
    ]

    # 2 binary labels for gender classification: man or woman
    labels = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    # Represent labels in one-hot format for binary classifier training,
    # BCEWithLogitsLoss requires target to have same shape as input
    # labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    train_files = [{"img": img, "label": label} for img, label in zip(images[:10], labels[:10])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-10:], labels[-10:])]

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
            EnsureTyped(keys=["img"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            AddChanneld(keys=["img"]),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            EnsureTyped(keys=["img"]),
        ]
    )
    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    auc_metric = ROCAUCMetric()

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    max_epochs = 5
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                if acc_metric > best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

if __name__ == '__main__':
    main()