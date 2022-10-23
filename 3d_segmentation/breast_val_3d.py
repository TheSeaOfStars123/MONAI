'''
  @ Date: 2022/6/21 16:14
  @ Author: Zhao YaChen
'''
import torch

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, EnsureTyped
from networks.net_factory_3d import net_factory_3d
from breast_segmentation_3d import get_file_list
'''
Check best model output with the input image and label
'''
def check_best_model():
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if pin_memory else "cpu")

    # get files
    train_files, val_files = get_file_list()

    # val transform
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
            EnsureTyped(keys=["dceph3", "label"]),
        ]
    )

    # val dataloader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # load model
    model_filename = "best_breast_unetmodel.pth"
    model = net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2)
    model.load_state_dict(torch.load(model_filename))
    model.eval()

    with torch.no_grad():








if __name__ == '__main__':
    check_best_model()