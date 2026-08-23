"""
带通滤波器工具 — 用于脚步声信号预处理
"""

import numpy as np
from scipy.signal import butter, sosfilt


def create_bandpass(
    low: float, high: float, fs: float, order: int = 4
) -> np.ndarray:
    """
    创建 Butterworth 带通滤波器 (SOS 格式, 数值稳定)。

    Args:
        low: 低频截止 (Hz)
        high: 高频截止 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数

    Returns:
        SOS 格式的滤波器系数数组
    """
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sos


def apply_bandpass(data: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """
    应用带通滤波器。

    Args:
        data: 输入信号 (帧数,) 或 (声道, 帧数)
        sos: SOS 格式的滤波器系数

    Returns:
        滤波后信号，形状与输入相同
    """
    return sosfilt(sos, data)


def energy_in_band(
    spectrum: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
) -> float:
    """
    计算频谱中指定频段的能量。

    Args:
        spectrum: FFT 幅度谱 (正频率部分)
        low_hz: 频段下限 (Hz)
        high_hz: 频段上限 (Hz)
        fs: 采样率

    Returns:
        频段内能量
    """
    n = len(spectrum)
    freqs = np.arange(n) * fs / (2 * n)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spectrum[mask] ** 2))
