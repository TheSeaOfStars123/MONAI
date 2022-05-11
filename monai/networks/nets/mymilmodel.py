'''
  @ Date: 2022/4/24 15:56
  @ Author: Zhao YaChen
'''
from typing import Dict, Optional, Union, cast
import torch
import torch.nn as nn

from monai.networks.nets import Densenet121
from monai.networks.nets.resnet import resnet50
from monai.utils.module import optional_import

# models, _ = optional_import("torchvision.models")

class MYMILModel(nn.Module):
    """
    My Multiple Instance Learning (MIL) model, for Breast Dataset
    """
    def __init__(
        self,
        num_classes: int,
        mil_mode: str = "att",
        pretrained: bool = True,
        backbone: Optional[Union[str, nn.Module]] = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive: " + str(num_classes))

        if mil_mode.lower() not in ["att"]:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.mil_mode = mil_mode.lower()
        self.attention = nn.Sequential()
        self.transformer = None  # type: Optional[nn.Module]

        if backbone is None:
            net = resnet50(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)
            # net = Densenet121(pretrained=False, spatial_dims=3, in_channels=1, out_channels=2)
            nfc = net.fc.in_features  # save the number of final features
            net.fc = torch.nn.Identity()  # remove final linear layer

            self.extra_outputs = {}  # type: Dict[str, torch.Tensor]

        else:
            raise ValueError("Unsupported backbone")
        if self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))
        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.net = net

    def calc_head(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        if self.mil_mode == "att":
            a = self.attention(x)  # a(1, 3, 1)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:
        # x(1, 3, 256, 256, 64)
        x = x[:, :, None, :, :, :]  # (1, 3, 1, 256, 256, 64)
        sh = x.shape
        x = x.reshape(sh[0]*sh[1], sh[2], sh[3], sh[4], sh[5])  # (3, 1, 256, 256, 64)

        x = self.net(x)  # (3, 2048)
        x = x.reshape(sh[0], sh[1], -1)  # (1, 3, 2048)

        if not no_head:
            x = self.calc_head(x)

        return x