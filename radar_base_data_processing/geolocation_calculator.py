import numpy as np
from geopy.distance import geodesic
from geopy.point import Point


def generate_lon_lat_1d_array(
        site_lon: float, site_lat: float,
        lon_interval: float, lat_interval: float,
        grid_h: int, grid_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成一维的经纬度序列，用于及数据插值
    """

    assert grid_h // 2 != 0, "grid_h must be odd"
    assert grid_w // 2 != 0, "grid_w must be odd"

    # 插值区域的经纬度序列
    interp_lon_array = (np.arange(-grid_w // 2, grid_w // 2) + 0.5) * lon_interval + site_lon
    interp_lat_array = (np.arange(-grid_h // 2, grid_h // 2) + 0.5)[::-1] * lat_interval + site_lat

    return interp_lon_array.astype(np.float32), interp_lat_array.astype(np.float32)  # 维度从高到低


def calculate_lon_interval(site_lat: float, distance_resolution: float = 0.25) -> float:
    """
    WGS-84标准下，计算指定纬度的经度间隔（0.25km/格的情况下，跨越了多少度）
    """

    lat_rad = np.deg2rad(site_lat)  # 纬度，弧度制

    a = 6378137  # 地球长轴半径，m
    b = 6356752.3142  # 地球短轴半径，m
    r = a * b / np.sqrt(a ** 2 * np.tan(lat_rad) ** 2 + b ** 2) / 1000  # 纬线圈半径，km

    circle_length = 2 * np.pi * r  # 纬线圈长，km

    return float(distance_resolution / circle_length * 360)


def calculate_lat_interval(site_lat: float, distance_resolution: float = 0.25) -> float:
    """
    计算指定纬度的纬度间隔（0.25km/格的情况下，跨越了多少度）
    """

    _, dst_lat = azi_dis_to_lon_lat(start_lon=0, start_lat=site_lat, azi_deg=0, dis_km=distance_resolution)
    delta1 = abs(dst_lat - site_lat)

    new_lat = site_lat - delta1 / 2  # 向下移动一半距离，再算一遍
    _, dst_lat = azi_dis_to_lon_lat(start_lon=0, start_lat=new_lat, azi_deg=0, dis_km=distance_resolution)

    return float(abs(dst_lat - new_lat))


def azi_dis_to_lon_lat(
        start_lon: float = 0, start_lat: float = 0,
        azi_deg: float = 0, dis_km: float = 0
) -> tuple[float, float]:
    """
    根据起始点的经纬度、方位角和距离，计算目标点的经纬度
    """

    # 创建起始点
    start_point = Point(start_lat, start_lon)

    # 使用geodesic函数计算目标点的经纬度
    destination_point = geodesic(kilometers=dis_km).destination(start_point, azi_deg)

    # 提取目标点的经纬度
    destination_latitude = destination_point.latitude
    destination_longitude = destination_point.longitude

    return destination_longitude, destination_latitude
