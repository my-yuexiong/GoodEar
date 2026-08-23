"""
GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

估算两个信号之间的到达时间差 (TDOA)。
PHAT 加权对混响和幅度变化具有鲁棒性，非常适合游戏音频场景。
"""

import numpy as np


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: float = 48000,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray]:
    """
    GCC-PHAT 广义互相关 - 相位变换。

    估算 sig 与 refsig 之间的时间延迟。
    正值表示 sig 领先 refsig（sig 先到达）。

    Args:
        sig: 信号 1 (左声道窗口)
        refsig: 信号 2 (右声道窗口)
        fs: 采样率 (Hz)
        max_tau: 最大搜索时延 (秒)，None = 全范围
        interp: 插值倍数，提高时延精度 (16 倍 ≈ 1.3μs @ 48kHz)

    Returns:
        (tau, cc)
        - tau: 估算时延 (秒)，正值=sig 领先 refsig
        - cc: 插值后的互相关函数

    原理:
        1. 计算两个信号的互功率谱: R = FFT(sig) * conj(FFT(refsig))
        2. PHAT 加权: R_phat = R / |R|  (只保留相位信息)
        3. 反 FFT 得到互相关函数
        4. 峰值位置 = 时延估计
    """
    # 计算 FFT 长度 (零填充以获得插值效果)
    n = len(sig) + len(refsig)

    # FFT
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)

    # 互功率谱
    R = SIG * np.conj(REFSIG)

    # PHAT 加权: 归一化幅度，只保留相位
    R_phat = R / (np.abs(R) + 1e-15)

    # 反 FFT (插值提高时延精度)
    cc = np.fft.irfft(R_phat, n=interp * n)

    # 限制搜索范围
    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    # 将负时延部分移到前面 (fftshift 等效)
    cc_centered = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # 找峰值
    peak_idx = int(np.argmax(np.abs(cc_centered)))
    shift = peak_idx - max_shift
    tau = shift / (interp * fs)

    return float(tau), cc_centered


def compute_itd(
    left: np.ndarray,
    right: np.ndarray,
    fs: float = 48000,
    max_tau: float = 0.00065,
    interp: int = 16,
) -> float:
    """
    计算左右声道的到达时间差 (ITD)。

    Args:
        left: 左声道信号
        right: 右声道信号
        fs: 采样率 (Hz)
        max_tau: 最大搜索时延 (秒)
        interp: 插值倍数

    Returns:
        ITD (秒): 正值 = 声音先到左耳 (声音靠左)
    """
    tau, _ = gcc_phat(left, right, fs=fs, max_tau=max_tau, interp=interp)
    return tau


def compute_ild(left: np.ndarray, right: np.ndarray) -> float:
    """
    计算左右声道的音量差 (ILD)。

    Args:
        left: 左声道信号
        right: 右声道信号

    Returns:
        ILD (dB): 正值 = 左声道更响 (声音靠左)
    """
    rms_l = float(np.sqrt(np.mean(left ** 2)) + 1e-10)
    rms_r = float(np.sqrt(np.mean(right ** 2)) + 1e-10)
    ild = 20.0 * np.log10(rms_l / rms_r)
    return float(ild)
