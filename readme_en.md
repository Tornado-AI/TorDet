# 🌪️ TorDet: A Refined Two-Stage Deep Learning Approach for Radar-Based Tornado Detection

------

## 1. Introduction

This repository provides the inference interface of the TorDet algorithm, along with a running example, designed for CINRAD S-band operational radars.

Paper: TGRS-2025-02627 [major revision]

## 2. Data Flow and Interface

**Data flow:**
 `.npz` preprocessed file → algorithm interface → output results

During development and integration, mainly refer to and modify the following two files:

- `debug_pipeline.py` (located in the main project directory)
- `TornadoDetector.detect()` method in `tordet/detector.py`

**Additional development requirements:**

- Process radar base data following the sample data structure
- Adapt the output data structure as needed (currently only printed output)

------

## 3. Environment Setup

The project depends on five core packages.
 Below are the development and validation environments:

| Dependency    | Development Env | Verified New Env |
| ------------- | --------------- | ---------------- |
| Python        | 3.10.14         | 3.13.7           |
| Torch         | 2.4.1+cu118     | 2.9.0+cu130      |
| SciPy         | 1.13.1          | 1.16.1           |
| Timm          | 0.9.7           | 1.0.20           |
| Geopy         | 2.4.1           | 2.4.1            |
| OpenCV-Python | 4.10.0.84       | 4.12.0.88        |

> ⚙️ Verified: upgrading dependencies does not affect algorithm performance; adjustments can be made based on device conditions.

------

## 4. Base Data Preprocessing `.npz` File Guide

The `.npz` file fed into the model must follow this structure:

```python
{
    "site_id": str,
    "utc_time": datetime.datetime,
    "grid_data": {
        "elev_0": {
            "ref_data_qc": np.ndarray,  # (1152, 1152)
            "vel_data_qc": np.ndarray,  # (1152, 1152)
            "sw_data_qc": np.ndarray    # (1152, 1152)
        },
        "elev_1": {...},
        "elev_2": {...},
        "longitude": np.ndarray,  # (1152,)
        "latitude": np.ndarray    # (1152,) from high to low latitude, descending order
    }
}
```

**Example reference:**
 `./temp_npz/Z_RADR_I_Z9200_20240427065400_O_DOR_SAD_CAP_FMT.bin.bz2.npz`

**Notes:**

- Base data should cover a **150 km radius** within the lowest **3 elevation angles** (default VCP21: 0.5°, 1.45°, 2.4°)

- Extracted variables include:

  - Reflectivity factor `ref`
  - Radial velocity `vel`
  - Spectrum width `sw`

- Model input must be interpolated onto a **250 m resolution uniform Cartesian grid**.
   You can refer to the template and utility functions in `radar_base_data_processing/base_data_io_templete.py` for adaptation.

- Missing value handling and range trimming:

  Before interpolation, missing values and velocity folding regions should be filled with NaN.

  After interpolation, variables should be processed as follows:

  - Reflectivity `ref`: values below 0 are set to 0; missing values set to 0
  - Radial velocity `vel`: missing values set to -999 (automatically handled by the model)
  - Spectrum width `sw`: values below 0 set to 0; missing values set to 0

- Additional quality control algorithms (e.g., clutter removal, de-aliasing) can be incorporated during preprocessing.

------

## 5. Output Format

Each base data file outputs a `list[dict]` structure, where each dict represents a detection result.

**Example output:**

```python
[
    {
        'latitude': 23.340631131942455,
        'longitude': 113.41293231565395,
        'site': 'Z9200',
        'utc_time': '20240427065400',
        'x': 599.1183431952662,   # x coordinate in interpolated 1152×1152 grid
        'y': 426.3313609467456    # y coordinate in interpolated 1152×1152 grid (latitude descending order)
    }
]
```
