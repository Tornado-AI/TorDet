import numpy as np

from radar_base_data_processing.geolocation_calculator import (
    calculate_lat_interval,
    calculate_lon_interval,
    generate_lon_lat_1d_array
)
from radar_base_data_processing.interpolations import interpolate_nearest


#######################################################################################################
# 由于各雷达基数据格式存在差异，这里仅提供插值操作伪代码，供在自有工程中调用既有函数时参考。
# 一些细节处理提供了代码，建议直接调用（radar_base_data_processing.geolocation_calculator）
#######################################################################################################

# === 无效值映射表，用于 np.nan_to_num() 处理 ===
nan_dict = {
    'REF': 0,        # 反射率：0 表示弱回波，可视为最低值
    'VEL': -999,     # 径向速度：-999 表示无效（有效范围约 -45 ~ 45）
    'SW': 0,         # 谱宽：0 表示无变化，可视为最低值
}

# === 产品对应的坐标网格映射关系 ===
mesh_dict = {
    "ref": "ref",
    "vel": "vel",
    "sw": "vel",     # SW 使用 VEL 的坐标网格
}

# === 默认配置（不建议修改） ===
grid_h = grid_w = 1152        # 网格尺寸（高度、宽度）
num_elevations = 3            # 仰角层数
products: list[str] = [       # 产品字段名（按项目定义）
    "ref_data_qc",
    "vel_data_qc",
    "sw_data_qc",
]

# === 雷达站点信息（示例值，请根据实际情况修改） ===
site_lon = 120.0
site_lat = 30.0
lon_interval = calculate_lon_interval(site_lat)
lat_interval = calculate_lat_interval(site_lat)


#######################################################################################################
# 主函数：最近邻插值，将极坐标雷达数据插值到规则经纬度格点
# 请结合具体基数据中的内容进行对应提取
#######################################################################################################
def grid_data_nearest(polar_data: dict) -> dict:
    """
    将极坐标雷达数据插值到规则经纬度格点（使用最近邻法）

    主要流程：
        1. 根据站点经纬度和格点间距生成目标网格；
        2. 对每个仰角层、每种产品执行最近邻插值；
        3. 替换 NaN 值；
        4. 添加经纬度序列；
        5. 返回网格化结果。

    参数：
        polar_data : dict
            原始极坐标雷达数据，第一层放仰角，第二层是具体的雷达变量极坐标下的数据

    返回：
        dict
            包含各仰角层插值结果和目标网格经纬度序列的字典。
            结构示例：
                {
                    "elev_0": {
                        "ref_data_qc": np.ndarray, (1152,1152)
                        "vel_data_qc": np.ndarray, (1152,1152)
                        "sw_data_qc": np.ndarray, (1152,1152)
                    },
                    "elev_1":...,
                    "elev_2":...,
                    "longitude": np.ndarray, (1152,)
                    "latitude": np.ndarray, (1152,)  # 从北纬高纬度到低纬度，降序序列
                }
    """

    # === 1. 生成目标网格 ===
    lon_dst_array, lat_dst_array = generate_lon_lat_1d_array(
        site_lon, site_lat, lon_interval, lat_interval, grid_h, grid_w
    )
    lon_mesh_dst, lat_mesh_dst = np.meshgrid(lon_dst_array, lat_dst_array)

    grid_data: dict = {}

    # === 2. 开始插值 ===
    for elev in range(num_elevations):
        grid_data[f"elev_{elev}"] = {}

        for product in products:
            product_name = product.split('_')[0]  # e.g. "ref", "vel", "sw"

            grid_data[f"elev_{elev}"][product] = interpolate_nearest(
                data=polar_data[f"elev_{elev}"][product],
                x_mesh_src=polar_data[f"elev_{elev}"][f"{mesh_dict[product_name]}_lon_mesh"],
                y_mesh_src=polar_data[f"elev_{elev}"][f"{mesh_dict[product_name]}_lat_mesh"],
                x_mesh_dst=lon_mesh_dst,
                y_mesh_dst=lat_mesh_dst
            )

    # === 3. 替换 NaN 值 ===
    for elev in range(num_elevations):
        for product in products:
            pname = product.split('_')[0].upper()
            grid_data[f"elev_{elev}"][product] = np.nan_to_num(
                grid_data[f"elev_{elev}"][product],
                nan=nan_dict[pname]
            )

    # === 4. 添加经纬度序列 ===
    grid_data["longitude"] = lon_dst_array
    grid_data["latitude"] = lat_dst_array

    # === 5. 返回结果 ===
    return grid_data
