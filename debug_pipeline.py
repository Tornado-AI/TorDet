from pathlib import Path
from pprint import pprint

import numpy as np

from tordet.detector import TornadoDetector


def load_preprocessed_npz(path: Path) -> dict:
    """
    加载 .npz 文件并转换为 dict 结构，便于后续处理
    结构示例：

    "site_id": str,
    "utc_time": datetime.datetime
    "grid_data": {
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

    该字典结构可以直接传入 detector.detect
    """
    with np.load(path, allow_pickle=True) as f:
        return {
            'site_id': f["site_id"].item(),
            'utc_time': f["utc_time"].item(),
            'grid_data': f["grid_data"].item()
        }


if __name__ == '__main__':
    device: str = 'cuda:0'

    detector = TornadoDetector(device)

    temp_npz = Path(r"temp_npz/Z_RADR_I_Z9200_20240427065400_O_DOR_SAD_CAP_FMT.bin.bz2.npz")

    item: dict = load_preprocessed_npz(temp_npz)

    detect_result = detector.detect(item)

    pprint(detect_result)

    '''
    示例输出 格式为 list[dict]
    每一个 dict 表示在一帧中的一处检测结果
    [
        { 
            'latitude': 23.340631131942455,
            'longitude': 113.41293231565395,
            'site': 'Z9200',
            'utc_time': '20240427065400',
            'x': 599.1183431952662,  # 在插值后的1152*1152数据的x坐标
            'y': 426.3313609467456
        }
    ]  
    
    '''
