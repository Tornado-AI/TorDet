# -*- coding: utf-8 -*-
"""
Author: 王茂宇
Description: 提取检测结果中的龙卷目标
"""

import cv2

import numpy as np


def extract_key_connected_components(data: np.array, binary_threshold: float = 0.5) -> np.ndarray:
    """
    提取图像中的有效连通区域，返回一个多通道矩阵，每个通道对应一个有效的连通区域
    在使用该算法前，必须确保图像适合使用该算法，满足其他设定要求
    """

    # 将图像二值化，大于阈值的部分设为1，小于阈值的部分设为0
    assert data.ndim == 2
    binary_image = (data > binary_threshold).astype(np.uint8)

    # 使用OpenCV的connectedComponentsWithStats函数找到二值化图像中的连通区域
    # labels是一个和原图像同样大小的矩阵，每个连通区域的所有像素都被标记为一个唯一的标签
    # stats是一个矩阵，每一行对应一个标签，包含该连通区域的信息（左上角x坐标，左上角y坐标，宽度，高度，面积）
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    # 过滤掉背景
    object_labels = np.arange(1, stats.shape[0])

    # 创建一个新的多通道矩阵，每个通道对应一个有效的连通区域
    labeled_channels = np.zeros((len(object_labels), *binary_image.shape), dtype=np.uint8)

    # 将每个有效的连通区域分配到一个单独的通道
    for idx, label in enumerate(object_labels):
        labeled_channels[idx][labels == label] = 1

    # 返回多通道矩阵，每个通道对应一个有效的连通区域
    return labeled_channels  # (num_components, h, w)


def calculate_keypoints_coordinates_xy(labeled_channels: np.ndarray, max_percent: float = 80.) -> np.ndarray:
    """
    计算每个连通区域的坐标均值, 使用最大20%的像素，计算平均坐标，返回坐标数组 [k, 2]
    """

    assert labeled_channels.ndim == 3

    # 记录结果
    key_points_xy = []

    # 计算每个连通区域的坐标均值
    for channel in labeled_channels:

        if channel.sum() < 20:
            continue

        # 找到值最大的20%的像素
        top_percent = np.percentile(channel, max_percent)

        # 找到符合条件的像素的坐标
        top_coordinates = np.argwhere(channel > top_percent)

        # 计算坐标均值
        y, x = np.mean(top_coordinates, axis=0)

        # 记录结果
        key_points_xy.append([x, y])

    return np.array(key_points_xy)  # (num_components, 2)


def probability_map_to_coords(
        probability_map: np.ndarray,
        binary_threshold: float = 0.7,
        max_percent: float = 80.
) -> np.ndarray:
    """
    将概率图转换为坐标数组
    :param probability_map: 概率图
    :param max_percent: 使用最大百分比的像素
    :return: 坐标数组
    """
    # 提取有效连通区域
    labeled_channels = extract_key_connected_components(probability_map, binary_threshold)

    # 计算坐标
    core_coords_xy = calculate_keypoints_coordinates_xy(labeled_channels, max_percent)

    return core_coords_xy  # (num_components, 2)


def index_to_position(index: np.ndarray, array: np.ndarray) -> np.ndarray:
    """
    将经度索引转换为经度
    :param index: 索引  # (n,)
    :param array: 等差数组  # (1200, )
    """

    # 递增的等差数列，索引是一个浮点数，介于两个整数之间，需要插值
    diff = np.diff(array).mean()  # 右减左

    # 分离整数部分和小数部分
    idx_int = index.astype(int)
    idx_decimal = index - idx_int

    return array[idx_int] + idx_decimal * diff  # (n,)
