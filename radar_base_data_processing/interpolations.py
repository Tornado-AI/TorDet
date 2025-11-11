import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree


class KDResampler(object):
    """
    使用 KDTree 进行二维数据的最邻近插值，可以调用多核心进行加速
    源码来自 PY-CINRAD 库
    """

    def __init__(
            self, data: np.ndarray, x: np.ndarray, y: np.ndarray, roi: float = 0.02
    ):
        x_ravel = x.ravel()
        y_ravel = y.ravel()
        self.tree = KDTree(np.dstack((x_ravel, y_ravel))[0])
        self.data = data
        self.roi = roi

    def map_data(self, x_out: np.ndarray, y_out: np.ndarray) -> np.ndarray:
        out_coords = np.dstack((x_out.ravel(), y_out.ravel()))[0]
        _, indices = self.tree.query(out_coords, distance_upper_bound=self.roi, workers=-1)  # workers=-1 使用所有核心
        invalid_mask = indices == self.tree.n
        indices[invalid_mask] = 0
        data = self.data.ravel()[indices]
        data[invalid_mask] = np.nan
        return data.reshape(x_out.shape)


def interpolate_nearest(
        data: np.ndarray,
        x_mesh_src: np.ndarray, y_mesh_src: np.ndarray,
        x_mesh_dst: np.ndarray, y_mesh_dst: np.ndarray,
) -> np.ndarray:
    """
    二维最近邻插值接口（Nearest Neighbor Interpolation）

    本函数将源网格（通常为雷达极坐标或不规则坐标）上的二维数据 `data`，
    通过最近邻插值的方式映射到目标网格（通常为规则的经纬度格点）上。

    参数说明：
        data : np.ndarray
            源网格上的二维数据场，如反射率(ref)、径向速度(vel)等。
            数据形状应与 x_mesh_src / y_mesh_src 一致。
        x_mesh_src : np.ndarray
            源网格的 X 坐标（或经度网格）。
        y_mesh_src : np.ndarray
            源网格的 Y 坐标（或纬度网格）。
        x_mesh_dst : np.ndarray
            目标网格的 X 坐标（或经度网格）。
        y_mesh_dst : np.ndarray
            目标网格的 Y 坐标（或纬度网格）。

    返回值：
        np.ndarray
            插值到目标网格后的二维数据场，与 x_mesh_dst / y_mesh_dst 的形状一致。
    """
    kds = KDResampler(data.copy(), x_mesh_src, y_mesh_src)
    return kds.map_data(x_mesh_dst, y_mesh_dst)
