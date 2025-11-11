from functools import partial
from operator import itemgetter

import numpy as np
import timm
import torch
import torch.nn as nn

from tordet.utils.align_transform import align_xy_from_low_resolution_representation

unpack_align = itemgetter('center_xy_low', 'center_xy_high', 'bbox_xyxy')


class HRNet(nn.Module):
    def __init__(
            self,
            mode: str,
            in_chans: int
    ):
        super(HRNet, self).__init__()

        assert mode in ['W18', 'W32'], f"mode must be in ['W18', 'W32']"

        self.model = timm.create_model(
            f'hrnet_{mode.lower()}', pretrained=False, features_only=True, in_chans=in_chans
        )

        # 为了方便调用, 保存名字，保留后4个尺度的特征通道数和特征
        self.backbone_name = f"HRNet-{mode}"
        self.out_channels = self.model.feature_info.channels()[-4:]

    def forward(self, x):
        return self.model(x)[-4:]


def get_backbone(mode: str, in_chans: int):
    return HRNet(mode, in_chans)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HRNetHead(nn.Module):
    def __init__(self, in_feat_chans: list[int], out_chans: int):
        super(HRNetHead, self).__init__()
        self.in_chans = in_feat_chans[-1]
        self.out_chans = out_chans

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(in_feat_chans)
        feat_out_chans = [32, 64, 128, 1024]
        self.heads = nn.ModuleList(
            [nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=True) for in_chans in feat_out_chans]
        )
        self.head_name = "HRNet"

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels):
        head_block = BasicBlock
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x: list[torch.Tensor]):
        out_feats = []

        y = self.incre_modules[0](x[0])

        for i, downsample in enumerate(self.downsamp_modules):
            out_feats.append(y)
            y = self.incre_modules[i + 1](x[i + 1]) + downsample(y)

        y = self.final_layer(y)
        out_feats.append(y)

        y = [head(out_feats[i]) for i, head in enumerate(self.heads)]

        return y


def get_head(in_feat_chans: list[int], out_chans: int):
    return HRNetHead(in_feat_chans, out_chans)


class DetectionStage(nn.Module):
    def __init__(self, in_chans: int = 9, out_chans: int = 1, ckpt: str = None):
        super(DetectionStage, self).__init__()

        self.backbone = get_backbone("W18", in_chans)
        in_feat_chans: list[int] = self.backbone.out_channels
        self.head = get_head(in_feat_chans, out_chans)

        self.net_name = f"{self.backbone.backbone_name}_{self.head.head_name}Head"

        self.align = partial(
            align_xy_from_low_resolution_representation,
            up_factor=32,
            bbox_size=128
        )
        self.load_state_dict(torch.load(ckpt, weights_only=False))
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            x = self.head(x)
            return x

    def predict(self, x):
        with torch.no_grad():
            x = torch.sigmoid(self.forward(x)[-1])  # 取最深的特征图

            x = np.squeeze(x.detach().cpu().numpy())  # (h, w)
            aligned_result = self.align(x)
            center_xy_low, center_xy_high, bbox_xyxy = unpack_align(aligned_result)  # (m, 2), (m, 2), (m, 4)
            return bbox_xyxy  # (m, 4)


def get_detection_stage(ckpt: str = None):
    return DetectionStage(ckpt=ckpt)