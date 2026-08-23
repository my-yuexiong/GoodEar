"""
脚步声检测 — Spectral Flux onset detection + 自适应阈值

通过频谱通量 (Spectral Flux) 检测游戏音频中的瞬态事件（脚步声起振），
配合带通滤波和自适应阈值，过滤掉背景噪音、枪声和爆炸声。
"""

import numpy as np
from collections import deque

from src.config import (
    SAMPLE_RATE,
    BP_LOW,
    BP_HIGH,
    BP_ORDER,
    FFT_FRAME_SIZE,
    FFT_HOP_SIZE,
    FLUX_HISTORY_SIZE,
    FLUX_THRESHOLD_K,
    MIN_ONSET_FRAMES,
    GUNSHOT_CENTROID_MIN,
    FOOTSTEP_CENTROID_MAX,
    GUNSHOT_DURATION_MAX,
    EXPLOSION_DURATION_MIN,
    EXPLOSION_BASS_RATIO_MIN,
)
from src.processor.filter import create_bandpass, apply_bandpass, energy_in_band
from src.processor.direction import estimate_azimuth


class FootstepDetector:
    """
    Spectral Flux 脚步声检测器。

    算法流程:
        1. 立体声 → 单声道 (求平均)
        2. 带通滤波 (200-800Hz 脚步声核心频段)
        3. 滑动窗口 FFT → 频谱通量
        4. 自适应阈值判定 (median + k * MAD)
        5. 防抖过滤 (120ms 最小间隔)
        6. 声音类型分类 (脚步 vs 枪声 vs 爆炸)

    用法:
        detector = FootstepDetector(fs=48000)
        events = detector.process(stereo_data)  # (2, N)
    """

    def __init__(
        self,
        fs: float = SAMPLE_RATE,
        bp_low: float = BP_LOW,
        bp_high: float = BP_HIGH,
        bp_order: int = BP_ORDER,
        frame_size: int = FFT_FRAME_SIZE,
        hop_size: int = FFT_HOP_SIZE,
        flux_history_size: int = FLUX_HISTORY_SIZE,
        flux_k: float = FLUX_THRESHOLD_K,
        min_onset_frames: int = MIN_ONSET_FRAMES,
    ):
        self.fs = fs
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.flux_k = flux_k
        self.min_onset_frames = min_onset_frames

        # 带通滤波器 (脚步核心检测频段)
        self.sos = create_bandpass(bp_low, bp_high, fs, order=bp_order)

        # 状态
        self.prev_spectrum: np.ndarray | None = None
        self.flux_history: deque = deque(maxlen=flux_history_size)
        self.last_onset_frame: int = -min_onset_frames - 1
        self.frame_count: int = 0

        # Hanning 窗
        self._window = np.hanning(frame_size)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, stereo: np.ndarray) -> list[dict]:
        """
        处理一个立体声窗口，检测其中的脚步声事件。

        Args:
            stereo: (2, N) float32 立体声音频

        Returns:
            list of dict，每个 dict:
                {
                    'frame_idx': int,        # 检测帧索引
                    'time_sec': float,        # 检测时间 (秒)
                    'azimuth': float,         # 方位角 (-90 ~ +90)
                    'amplitude_db': float,    # RMS 能量 (dB)
                    'sound_type': str,        # 'footstep' | 'gunshot' | 'explosion' | 'unknown'
                }
        """
        if stereo.shape[1] < self.frame_size:
            return []

        # 合成立体声 → 单声道
        mono = (stereo[0] + stereo[1]) * 0.5

        # 带通滤波
        filtered = apply_bandpass(mono, self.sos)

        events = []
        n_frames = (len(filtered) - self.frame_size) // self.hop_size + 1

        for i in range(n_frames):
            start = i * self.hop_size
            frame = filtered[start : start + self.frame_size]

            # 加窗 + FFT
            windowed = frame * self._window
            spectrum = np.abs(np.fft.rfft(windowed))

            # Spectral Flux (只算正差值)
            if self.prev_spectrum is not None:
                diff = spectrum - self.prev_spectrum
                flux = float(np.sqrt(np.mean(np.maximum(diff, 0) ** 2)))
            else:
                flux = 0.0

            self.prev_spectrum = spectrum.copy()
            self.flux_history.append(flux)
            self.frame_count += 1

            # 需要积累一定的历史才能计算自适应阈值
            if len(self.flux_history) < 10:
                continue

            # 自适应阈值: median + k * MAD * 1.4826
            arr = np.array(list(self.flux_history))
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            threshold = med + self.flux_k * mad * 1.4826

            # Onset 检测 + 防抖
            frames_since_last = self.frame_count - self.last_onset_frame
            if flux > threshold and frames_since_last >= self.min_onset_frames:
                self.last_onset_frame = self.frame_count

                # 用原始立体声计算方位角
                left = stereo[0, start : start + self.frame_size]
                right = stereo[1, start : start + self.frame_size]
                azimuth = estimate_azimuth(left, right, fs=self.fs)

                # 用未滤波的原始信号计算能量
                raw_frame = mono[start : start + self.frame_size]
                rms = float(np.sqrt(np.mean(raw_frame ** 2)) + 1e-9)
                amp_db = float(20.0 * np.log10(rms))

                # 分类声音类型
                sound_type = classify_sound_type(spectrum, self.fs)

                events.append({
                    'frame_idx': self.frame_count,
                    'time_sec': self.frame_count * self.hop_size / self.fs,
                    'azimuth': azimuth,
                    'amplitude_db': amp_db,
                    'sound_type': sound_type,
                })

        return events

    def reset(self):
        """重置检测器状态。"""
        self.prev_spectrum = None
        self.flux_history.clear()
        self.last_onset_frame = -self.min_onset_frames - 1
        self.frame_count = 0


# ------------------------------------------------------------------
# 声音类型分类
# ------------------------------------------------------------------

def classify_sound_type(
    spectrum: np.ndarray,
    fs: float,
    centroid_gunshot_min: float = GUNSHOT_CENTROID_MIN,
    centroid_footstep_max: float = FOOTSTEP_CENTROID_MAX,
    explosion_bass_min: float = EXPLOSION_BASS_RATIO_MIN,
) -> str:
    """
    基于频谱特征区分声音类型。

    判定逻辑:
        - 爆炸: 低频(bass)占比 > 50%
        - 枪声: 频谱质心 > 3000Hz (高频瞬态)
        - 脚步: 质心在 200-2000Hz 之间，中频能量占比高

    Args:
        spectrum: FFT 幅度谱
        fs: 采样率

    Returns:
        'footstep' | 'gunshot' | 'explosion' | 'unknown'
    """
    n = len(spectrum)
    freqs = np.arange(n) * fs / (2 * n)
    total_energy = float(np.sum(spectrum ** 2)) + 1e-10

    # 频谱质心
    centroid = float(np.sum(freqs * (spectrum ** 2)) / total_energy)

    # 各频段能量占比
    bass_ratio = float(energy_in_band(spectrum, 60, 200, fs) / total_energy)
    mid_ratio = float(energy_in_band(spectrum, 200, 800, fs) / total_energy)
    high_ratio = float(energy_in_band(spectrum, 2000, 4000, fs) / total_energy)

    # 判定
    if bass_ratio > explosion_bass_min:
        return 'explosion'
    if centroid > centroid_gunshot_min:
        return 'gunshot'
    if mid_ratio > 0.2 or (high_ratio > 0.15 and centroid < centroid_footstep_max):
        return 'footstep'

    return 'unknown'
