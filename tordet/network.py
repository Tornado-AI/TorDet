import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from tordet.modules import get_detection_stage, get_positioning_stage


class TorDet(nn.Module):
    def __init__(self, stage_1: nn.Module, stage_2: nn.Module):
        super(TorDet, self).__init__()
        self.stage_1 = stage_1
        self.stage_2 = stage_2
        self.eval()

    def forward(self, x):
        results = []
        xyxy = self.stage_1.predict(x)  # (m, 4)  m 个 bbox

        # 从 xyxy 中提取 box 数据
        m = xyxy.shape[0]

        if m == 0:
            return []

        for box in xyxy:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            patch = x[:, :, y1:y2, x1:x2]
            patch_out = self.stage_2.predict(patch)

            if patch_out is None:
                continue

            if patch_out['center_xy'].size == 0:
                continue

            if 'cls' in patch_out:
                cls = patch_out['cls']  # 0,1,2
                if cls == 0:
                    continue

            center_xy = patch_out['center_xy']
            center_xy_global = center_xy + np.array([x1, y1])

            patch_out.update({'bbox': box, 'center_xy_global': center_xy_global})

            # 格式转换
            for k,v in patch_out.items():
                if isinstance(v, (np.floating, np.integer)):
                    patch_out[k] = v.item()


            results.append(patch_out)

        return results


def get_tordet(det_ckpt: str | Path, pos_ckpt: str | Path) -> nn.Module:
    """
    构建并返回一个 TorDet 模型实例。

    参数:
        det_ckpt (str | Path): 检测阶段模型的检查点路径，可以是字符串或 Path 对象。
        pos_ckpt (str | Path): 定位阶段模型的检查点路径，可以是字符串或 Path 对象。

    返回:
        nn.Module: 一个 TorDet 模型实例，包含检测阶段和定位阶段的子模块。
    """
    # 加载检测阶段模型
    det_stage = get_detection_stage(ckpt=det_ckpt)

    # 加载定位阶段模型
    pos_stage = get_positioning_stage(ckpt=pos_ckpt)

    # 使用加载的检测和定位阶段模型构建 TorDet 实例
    return TorDet(det_stage, pos_stage)



if __name__ == '__main__':
    det_ckpt = r'D:\Laboratory_Work\2024_Tornado_Independently\15_两阶段拼接_部署用\TorDet\ckpt\detection_stage.pth'
    pos_ckpt = r'D:\Laboratory_Work\2024_Tornado_Independently\15_两阶段拼接_部署用\TorDet\ckpt\positioning_stage.pth'

    net = get_tordet(det_ckpt=det_ckpt, pos_ckpt=pos_ckpt)

    # 读取雷达数据
    radar_data = torch.randn(1, 9, 1152, 1152)
    results = net(radar_data)

    print(results)
