import cv2
import numpy as np


def align_xy_from_low_resolution_representation(
        low_reso: np.ndarray,  # (1, 36, 36)
        up_factor: int = 32,
        bbox_size: int = 128,
) -> dict:
    """
        同时考虑样本集和实际应用时的情况，默认当作推理时的情况，并且可能有多个目标，需要用连通域分析
    """

    assert low_reso.ndim in [2, 3], f'low_reso 的维度应为 2 或 3，但实际为 {low_reso.ndim}'

    cls_pred: np.ndarray = low_reso[0] if low_reso.ndim == 3 else low_reso  # (36, 36)
    h_low = cls_pred.shape[0]  # 36
    h_high = h_low * up_factor  # 1152

    # 过滤背景
    cls_mask: np.ndarray[np.bool_] = cls_pred > 0.5  # (36, 36) # 警告忽略

    # 如果只有背景, 直接返回
    if cls_mask.sum() == 0:
        return {
            'center_xy_low': np.empty((0, 2), dtype=np.float32),
            'center_xy_high': np.empty((0, 2), dtype=np.float32),
            'bbox_xyxy': np.empty((0, 4), dtype=np.float32)
        }

    # 连通域分析 cv2.connectedComponents
    _, components_labels = cv2.connectedComponents(cls_mask.astype(np.uint8), connectivity=4)

    # 如果只有背景，直接返回
    if components_labels.max() == 0:
        return {
            'center_xy_low': np.empty((0, 2), dtype=np.float32),
            'center_xy_high': np.empty((0, 2), dtype=np.float32),
            'bbox_xyxy': np.empty((0, 4), dtype=np.float32)
        }

    # 逐个连通域计算坐标
    center_xy_low = []

    for i in range(1, components_labels.max() + 1):
        component_mask = components_labels == i

        # 获取全部的点的坐标，然后逐点遍历
        points = np.column_stack(np.where(component_mask))  # (n, 2) [y, x]

        # 提取对应位置的cls_pred,作为置信度
        confidence = cls_pred[component_mask]  # (n,)

        # 如果点数超过4个，只保留置信度最高的4个
        if len(points) > 4:
            top_4_idx = np.argsort(confidence)[-4:]
            # confidence = confidence[top_4_idx]
            points = points[top_4_idx]  # (4, 2)

        # 分别计算x和y的坐标，0.5*(xmax+xmin), 0.5*(ymax+ymin)
        x_center = 0.5 * (points[:, 1].max() + points[:, 1].min())
        y_center = 0.5 * (points[:, 0].max() + points[:, 0].min())

        center_xy_low.append((x_center, y_center))

    # 计算高分辨率的中心坐标
    center_xy_high = np.array(center_xy_low) * up_factor + (up_factor - 1) / 2
    # 根据bbox_size计算左上角坐标，考虑边界情况
    x1y1 = np.clip(np.round(center_xy_high - bbox_size / 2, 0), 0, h_high - bbox_size).astype(np.int32)
    x2y2 = x1y1 + bbox_size

    bbox_xyxy = np.column_stack([x1y1, x2y2])  # (n, 4)

    return {
        'center_xy_low': np.array(center_xy_low),  # (n, 2)
        'center_xy_high': center_xy_high,  # (n, 2)
        'bbox_xyxy': bbox_xyxy  # (n, 4)
    }
