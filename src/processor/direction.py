"""
方向估算 — ILD + ITD 融合

通过分析左右声道的音量差(ILD)和时间差(ITD)，
估算声源在水平面上的方位角 (-90° ~ +90°)。
"""

import math
import numpy as np

from src.config import (
    MAX_TAU,
    MAX_ILD_DB,
    ITD_WEIGHT,
    ILD_WEIGHT,
    GCC_INTERP,
    GCC_MAX_TAU,
)
from src.processor.gcc_phat import compute_itd, compute_ild


def ild_to_azimuth(ild_db: float, max_ild_db: float = MAX_ILD_DB) -> float:
    """
    将 ILD (dB) 映射为方位角。

    使用裁剪线性映射：+/−15dB → +/−90°

    Args:
        ild_db: ILD 值 (dB)，正=左更响
        max_ild_db: 声道极值 dB

    Returns:
        方位角 (度): -90 (极右) ~ +90 (极左)
    """
    ratio = np.clip(ild_db / max_ild_db, -1.0, 1.0)
    return float(ratio * 90.0)


def itd_to_azimuth(tau: float, max_tau: float = MAX_TAU) -> float:
    """
    将 ITD (秒) 映射为方位角。

    使用球头模型: azimuth = arcsin(tau / max_tau)

    Args:
        tau: 到达时间差 (秒)，正=左先到
        max_tau: 最大可能时延 (秒)

    Returns:
        方位角 (度): -90 (极右) ~ +90 (极左)
    """
    ratio = np.clip(tau / max_tau, -1.0, 1.0)
    return float(np.arcsin(ratio) * 180.0 / np.pi)


def estimate_azimuth(
    left: np.ndarray,
    right: np.ndarray,
    fs: float = 48000,
    max_tau: float = MAX_TAU,
    max_ild_db: float = MAX_ILD_DB,
    itd_weight: float = ITD_WEIGHT,
    ild_weight: float = ILD_WEIGHT,
) -> float:
    """
    融合 ILD 和 ITD 估算方位角。

    返回正角度 = 声音偏向左侧（0 = 正前方）。
    纯立体声无法区分前后（混淆锥），所有结果映射到前方半球。

    Args:
        left: 左声道信号 (1D float32 数组)
        right: 右声道信号 (1D float32 数组)
        fs: 采样率
        max_tau: 最大 ITD
        max_ild_db: 最大 ILD
        itd_weight: ITD 权重 (默认 0.6)
        ild_weight: ILD 权重 (默认 0.4)

    Returns:
        方位角 (度): -90 ~ +90，0 = 正前方，+ = 左侧
    """
    # ILD 估算
    ild_db = compute_ild(left, right)
    az_ild = ild_to_azimuth(ild_db, max_ild_db)

    # ITD 估算 (GCC-PHAT)
    tau = compute_itd(left, right, fs=fs, max_tau=max_tau, interp=GCC_INTERP)
    az_itd = itd_to_azimuth(tau, max_tau)

    # 加权融合
    azimuth = itd_weight * az_itd + ild_weight * az_ild
    azimuth = float(np.clip(azimuth, -90.0, 90.0))

    return azimuth


def estimate_azimuth_with_confidence(
    left: np.ndarray,
    right: np.ndarray,
    fs: float = 48000,
) -> tuple[float, float]:
    """
    估算方位角并返回置信度。

    置信度基于 ILD 和 ITD 估算的一致性。
    两个估算越接近，置信度越高。

    Returns:
        (azimuth_deg, confidence_0_to_1)
    """
    ild_db = compute_ild(left, right)
    az_ild = ild_to_azimuth(ild_db)

    tau = compute_itd(left, right, fs=fs)
    az_itd = itd_to_azimuth(tau)

    azimuth = ITD_WEIGHT * az_itd + ILD_WEIGHT * az_ild
    azimuth = float(np.clip(azimuth, -90.0, 90.0))

    # 一致性 = 1 - normalized_diff
    diff = abs(az_ild - az_itd)
    confidence = float(np.clip(1.0 - diff / 90.0, 0.0, 1.0))

    return azimuth, confidence
