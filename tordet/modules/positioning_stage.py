import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tordet.utils.extractor import probability_map_to_coords


class CBAM(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        super(CBAM, self).__init__()

        # 通道注意力模块
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channel, in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_channel // ratio, in_channel, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力模块
        padding = kernel_size // 2
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力机制
        b, c, h, w = x.shape

        max_pool = self.max_pool(x).view([b, c])  # 全局最大池化并调整形状
        avg_pool = self.avg_pool(x).view([b, c])  # 全局平均池化并调整形状

        # 通道注意力的两条路径
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))

        # 两条路径相加，并通过sigmoid归一化
        channel_attention = self.sigmoid_channel(max_out + avg_out).view([b, c, 1, 1])
        x = x * channel_attention  # 与输入特征图相乘

        # 空间注意力机制
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度上的最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度上的平均池化
        spatial_attention = torch.cat([max_out, avg_out], dim=1)  # 拼接
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(spatial_attention))  # 卷积后归一化
        x = x * spatial_attention  # 与输入特征图相乘

        return x


# Basic convolution module for encoding and decoding
class DownDoubleConv(nn.Module):
    """(conv=>BN=>ReLu)*2"""

    def __init__(self, in_ch, out_ch, mode='Vanilla'):

        assert mode in ['Vanilla', 'CBAM', 'SCSE'], 'mode must be in ["Vanilla", "CBAM", "SCSE"]'

        super(DownDoubleConv, self).__init__()

        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        match mode:
            case "CBAM":
                self.attention = CBAM(out_ch)
            case _:
                pass

    def forward(self, x):

        match self.mode:
            case "CBAM":
                x = self.conv(x)
                x = self.attention(x)
            case "SCSE":
                x = self.conv(x)
                x = self.attention(x)
            case _:
                x = self.conv(x)

        return x


class UpDoubleConv(nn.Module):
    """(conv=>BN=>ReLu)*2"""

    def __init__(self, in_ch, out_ch):
        super(UpDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# Network input convolution module
class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = UpDoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


# Network output convolution module
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = ChangeChannels(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)

        return x


# down-sampling module in encoder
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, mode: str = "Vanilla"):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DownDoubleConv(in_ch, out_ch, mode)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


# up-sampling module in decoder
# include Linear interpolation、Deconvolution and sub_pixel
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, in_ch_x1, r=2):
        super(Up, self).__init__()
        tmp_out_ch = in_ch_x1 * r * r
        self.up = nn.Sequential(
            ChangeChannels(in_ch_x1, tmp_out_ch),
            nn.PixelShuffle(r))

        self.conv = UpDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 不使用特殊尺寸时的补齐尺寸
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        #  将x1填充到x2的尺寸
        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


# Module used to change channel
class ChangeChannels(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChangeChannels, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes=1, base_channel=64, mode: str = "Vanilla"):
        """
            :param in_channels: Number of input data channels
            :param n_classes: Number of output data channels
        """
        super(UNet, self).__init__()
        # Build encoding module
        self.inc = InConv(in_channels, base_channel)  # base_channel
        self.down1 = Down(base_channel, base_channel * 2, mode)  # base_channel * 2
        self.down2 = Down(base_channel * 2, base_channel * 4, mode)  # base_channel * 4
        self.down3 = Down(base_channel * 4, base_channel * 8, mode)  # base_channel * 8
        self.down4 = Down(base_channel * 8, base_channel * 8, mode)  # base_channel * 8

        # Build decoding module
        self.up1 = Up(base_channel * 16, base_channel * 4, base_channel * 8)  # base_channel * 4
        self.up2 = Up(base_channel * 8, base_channel * 2, base_channel * 4)  # base_channel * 2
        self.up3 = Up(base_channel * 4, base_channel, base_channel * 2)  # base_channel
        self.up4 = Up(base_channel * 2, base_channel, base_channel)  # base_channel
        self.outc = OutConv(base_channel, n_classes)

        self.out_channels = [base_channel, base_channel, base_channel * 2, base_channel * 4, base_channel * 8]
        self.backbone_name = f'UNet-{mode}'

    # Forward propagation of model
    def forward(self, x):
        enc1 = self.inc(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        enc5 = self.down4(enc4)

        decode_features = [enc5]

        dec1 = self.up1(enc5, enc4)
        decode_features.append(dec1)
        del enc5, enc4

        dec2 = self.up2(dec1, enc3)
        decode_features.append(dec2)
        del enc3

        dec3 = self.up3(dec2, enc2)
        decode_features.append(dec3)
        del enc2

        dec4 = self.up4(dec3, enc1)
        decode_features.append(dec4)
        del enc1

        # 倒序输出
        decode_features.reverse()

        return decode_features


def get_backbone(mode: str = "CBAM", in_channels=9):
    config_dict = {
        "CBAM": {"base_channel": 64, "mode": "CBAM"},
    }

    config = config_dict[mode]
    net = UNet(**config, in_channels=in_channels)

    net.model_info = {
        'backbone_name': 'UNet',
        'mode': mode,
    }

    return net


class DirectHead(nn.Module):
    def __init__(self, in_feat_chans: list[int]):
        super(DirectHead, self).__init__()
        self.in_chans = in_feat_chans

        self.center_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_chans[i], 1, 1, bias=True),
                nn.Sigmoid()
            ) for i in range(4)
        ])

        self.pair_head = nn.Sequential(
            nn.Conv2d(self.in_chans[0], 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_chans[-1], 3)
        )

        self.head_name = "Direct"

    def forward(self, x: list[torch.Tensor]) -> dict:
        center_hm = [head(x[i]) for i, head in enumerate(self.center_heads)]
        pair_hm = self.pair_head(x[0])
        cls_logits = self.cls_head(x[-1])

        return {
            'center_hm': center_hm,
            'pair_hm': pair_hm,
            'cls_logits': cls_logits
        }


def get_head(in_feat_chans: list[int]):
    return DirectHead(in_feat_chans)


class PositioningStage(nn.Module):
    def __init__(self, in_chans: int = 9, ckpt=None):
        super(PositioningStage, self).__init__()

        self.backbone = get_backbone(in_channels=in_chans)
        in_feat_chans: list[int] = self.backbone.out_channels
        self.head = get_head(in_feat_chans)

        self.net_name = f"{self.backbone.backbone_name}_{self.head.head_name}Head"

        if ckpt is not None:
            self.load_state_dict(torch.load(ckpt, weights_only=False))

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def predict(self, x):
        # x(1, c, h, w)

        x: dict = self.forward(x)

        # 分析输出
        center_hm = torch.sigmoid(x['center_hm'][0])  # (1, 1, h, w)
        # pair_hm = torch.sigmoid(x['pair_hm'])  # (1, 1, h, w)
        cls_logits = torch.softmax(x['cls_logits'], dim=-1)  # (1, 3)

        center_hm = np.squeeze(center_hm.detach().cpu().numpy())  # (h, w)
        # pair_hm = np.squeeze(pair_hm.detach().cpu().numpy())

        cls = torch.argmax(cls_logits, dim=-1).item()  # 0, 1, 2

        if cls == 0:
            return None
        else:
            p0, p1, p2 = cls_logits[0].detach().cpu().numpy()
            p1_c = p1 / (p1 + p2)
            p2_c = p2 / (p1 + p2)

            center_xy = probability_map_to_coords(center_hm)  # (m, 2)
            return {
                'center_xy': center_xy,
                'cls': cls,
                'p0': p0,
                'p1': p1,
                'p2': p2,
                'p1_c': p1_c,
                'p2_c': p2_c
            }


def get_positioning_stage(ckpt=None):
    return PositioningStage(ckpt=ckpt)
