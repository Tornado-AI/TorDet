# -*- coding: utf-8 -*-
"""
Author: 王茂宇
Description: 雷达数据归一化，用于模型输入
"""

import numpy as np

import torch
import torch.nn as nn


class RefNormalize(nn.Module):
    """
    反射率归一化

    :param ref_max: 反射率最大值
    """

    def __init__(self, ref_max: float = 80):
        super().__init__()
        self.ref_max = ref_max

    def forward(self, ref: np.ndarray):
        """
        :param ref: 反射率数据
        :return: 归一化后的反射率数据
        """
        ref = ref / self.ref_max
        ref[ref < 0] = 0
        return torch.from_numpy(ref)


class VelNormalize(nn.Module):
    """
    速度归一化

    :param vel_max: 速度最大值
    """

    def __init__(self, vel_max: float = 45):
        super().__init__()
        self.vel_max = vel_max

    def forward(self, vel: np.ndarray):
        """
        :param vel: 速度数据
        :return: 归一化后的速度数据
        """
        vel[vel < -100] = 0
        vel = vel / self.vel_max
        return torch.from_numpy(vel)


class SWNormalize(nn.Module):
    """
    谱宽归一化

    :param sw_max: 谱宽最大值
    """

    def __init__(self, sw_max: float = 30):
        super().__init__()
        self.sw_max = sw_max

    def forward(self, sw: np.ndarray):
        """
        :param sw: 谱宽数据
        :return: 归一化后的谱宽数据
        """
        sw = sw / self.sw_max
        sw[sw < 0] = 0
        return torch.from_numpy(sw)
