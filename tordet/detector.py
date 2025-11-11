# 作者：王茂宇 中国海洋大学 wangmaoyu@stu.ouc.edu.cn
import datetime

import numpy as np
import torch

from tordet.data_transforms import RefNormalize, VelNormalize, SWNormalize
from tordet.network import get_tordet
from tordet.utils.extractor import index_to_position

"""
检测器接口，不要动，动了必后悔

作者：王茂宇 中国海洋大学 wangmaoyu@stu.ouc.edu.cn
"""


class TornadoDetector:
    det_ckpt = 'tordet/ckpt/detection_stage.pth'
    pos_ckpt = 'tordet/ckpt/positioning_stage.pth'

    args = {
        "in_channels": 9,  # 输入通道数
        "num_elevations": 3,  # 仰角层数
        "ref_norm_max": 80.0,  # 反射率归一化上限
        "vel_norm_max": 45.0,  # 速度归一化上限
        "sw_norm_max": 30.0  # 谱宽归一化上限
    }

    def __init__(
            self,
            device: str,
    ):

        # 输入通道数
        self.in_channels: int = self.args["in_channels"]

        # 设备
        self.device: torch.device = torch.device(device)

        # 模型
        self.detector = get_tordet(self.det_ckpt, self.pos_ckpt)
        self.detector.to(self.device)

        # 归一化器
        self.ref_norm = RefNormalize(self.args["ref_norm_max"])
        self.vel_norm = VelNormalize(self.args["vel_norm_max"])
        self.sw_norm = SWNormalize(self.args["sw_norm_max"])

    def normalize(self, ref_data, vel_data, sw_data) -> torch.Tensor:
        """
        组合、归一化数据，用于推理
        :param ref_data: 反射率数据
        :param vel_data: 速度数据
        :param sw_data: 谱宽数据
        :return: 组合、归一化后的数据
        """

        ref_data: torch.Tensor = self.ref_norm(ref_data)  # (3, 1152, 1152)
        vel_data: torch.Tensor = self.vel_norm(vel_data)  # (3, 1152, 1152)
        sw_data: torch.Tensor = self.sw_norm(sw_data)  # (3, 1152, 1152)

        data = torch.cat([ref_data, vel_data, sw_data], dim=0)  # (9, 1152, 1152)

        return data[None, ...]  # (1, 9, 1152, 1152)

    def detect(self, item: dict) -> list[dict]:
        """
        # 读取方式完全取决于预处理，以及模型本身的需要，代码需要随之变动
        """

        # 读取基础信息
        utc_time: datetime.datetime = item["utc_time"]
        site_id: str = item["site_id"]

        # 读取关键部分数据，按变量优先堆叠
        grid_data = item["grid_data"]

        # 经纬度序列
        lon = grid_data["longitude"]  # (1152,)
        lat = grid_data["latitude"]  # (1152,)

        elevations = range(self.args["num_elevations"])

        #    "grid_data": {
        #         "elev_0": {
        #             "ref_data_qc": np.ndarray, (1152,1152)
        #             "vel_data_qc": np.ndarray, (1152,1152)
        #             "sw_data_qc": np.ndarray, (1152,1152)
        #         },
        #         "elev_1":...,
        #         "elev_2":...,
        #         "longitude": np.ndarray, (1152,)
        #         "latitude": np.ndarray, (1152,)  # 从北纬高纬度到低纬度，降序序列
        #     }
        grid_data = {
            'ref_data': np.array([grid_data[f'elev_{i}']['ref_data_qc'] for i in elevations]),
            'vel_data': np.array([grid_data[f'elev_{i}']['vel_data_qc'] for i in elevations]),
            'sw_data': np.array([grid_data[f'elev_{i}']['sw_data_qc'] for i in elevations]),
        }

        # 归一化, 使用键值对传参，
        radar_data: torch.Tensor = self.normalize(**grid_data)  # (1, 9, 1152, 1152)
        # 9个通道 [ref_0, ref_1, ref_2, vel_0, vel_1, vel_2, sw_0, sw_1, sw_2]

        # 检测
        with torch.no_grad():
            radar_data = radar_data.to(self.device)
            predictions: list = self.detector(radar_data)

        # 更新一些信息
        for pred in predictions:
            x, y = pred['center_xy_global'][0]
            prob = 1 - float(pred['p0'])
            if prob >= 0.3:
                pred.update(
                    {
                        "site": site_id,  # 站点ID str
                        "utc_time": utc_time.strftime("%Y%m%d%H%M%S"),  # str
                        "x": float(x),  # 在 1152 * 1152 网格中的位置
                        "y": float(y),  # 在 1152 * 1152 网格中的位置，纬度是降序北纬
                        "longitude": float(index_to_position(x, lon)),  # 转换
                        "latitude": float(index_to_position(y, lat)),
                        "probability": float(pred['p2_c'])
                    }
                )


        # filter main info
        keys = ['site', 'utc_time', 'x', 'y', 'longitude', 'latitude']
        for i, pred in enumerate(predictions):
            predictions[i] = {k: v for k, v in pred.items() if k in keys}

        return predictions
