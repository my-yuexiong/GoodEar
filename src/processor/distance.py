"""
距离估算 — 基于 RMS 能量估算声源近似距离

注意: 此模块高度依赖游戏音频引擎的衰减模型，仅提供粗略估计。
不同游戏的衰减特征不同（Valorant 用 FLAT 衰减），需要校准。
"""

import numpy as np

from src.config import REF_AMP_DB, REF_DIST_M, MAX_DIST_M, DB_PER_DOUBLING


def estimate_distance(
    amplitude_db: float,
    ref_amp_db: float = REF_AMP_DB,
    ref_dist_m: float = REF_DIST_M,
    max_dist_m: float = MAX_DIST_M,
    db_per_doubling: float = DB_PER_DOUBLING,
) -> float:
    """
    基于 RMS 能量估算距离。

    简化模型: 距离增倍 ≈ 能量衰减 db_per_doubling dB

    Args:
        amplitude_db: 检测到的 RMS 能量 (dB)
        ref_amp_db: 参考距离处的 RMS 能量 (dB) — 需要校准
        ref_dist_m: 参考距离 (米)
        max_dist_m: 最大距离 (米)
        db_per_doubling: 距离翻倍对应的 dB 衰减

    Returns:
        估算距离 (米)，限制在 [1.0, max_dist_m]
    """
    db_diff = ref_amp_db - amplitude_db
    # 距离 = ref_dist * 2^(db_diff / db_per_doubling)
    multiplier = 2.0 ** (db_diff / db_per_doubling)
    distance = ref_dist_m * multiplier
    return float(np.clip(distance, 1.0, max_dist_m))


def distance_to_zone(distance_m: float) -> str:
    """
    将距离映射为区域标签。

    Args:
        distance_m: 距离 (米)

    Returns:
        'close' | 'medium' | 'far'
    """
    if distance_m < 10:
        return 'close'
    elif distance_m < 25:
        return 'medium'
    else:
        return 'far'


def auto_calibrate(
    rms_history: list[float],
    percentile: float = 90.0,
) -> float:
    """
    自动校准参考能量。

    取最近一段时间 RMS 的第 N 百分位数作为参考值。
    假设大部分脚步声在中等距离。

    Args:
        rms_history: RMS 能量历史 (dB)
        percentile: 使用的百分位数

    Returns:
        校准后的参考能量 (dB)
    """
    if len(rms_history) < 10:
        return REF_AMP_DB  # 不够数据，用默认值
    return float(np.percentile(rms_history, percentile))
