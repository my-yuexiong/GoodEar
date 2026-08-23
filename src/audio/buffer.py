"""
环形缓冲区 — 线程安全的音频数据缓存
"""

import threading
from collections import deque

import numpy as np


class RingBuffer:
    """
    线程安全的环形缓冲区，存储最近 N 个音频 chunk。

    用于 AudioCapture callback → 处理线程 之间的数据传递。
    """

    def __init__(self, max_chunks: int = 8):
        self._buffer: deque = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def append(self, stereo: np.ndarray):
        """
        追加一个音频 chunk。
        stereo shape: (2, frame_count), float32 [-1.0, 1.0]
        """
        with self._lock:
            self._buffer.append(stereo)

    def get_latest(self, n_frames: int | None = None) -> np.ndarray | None:
        """
        获取缓冲区中拼接后的最新音频数据。

        Args:
            n_frames: 返回最近 N 帧，None 则返回全部

        Returns:
            (2, N) float32 数组，或 None（缓冲区为空时）
        """
        with self._lock:
            if not self._buffer:
                return None
            chunks = list(self._buffer)

        data = np.concatenate(chunks, axis=1)
        if n_frames is not None and data.shape[1] > n_frames:
            data = data[:, -n_frames:]
        return data

    def ready_for(self, n_frames: int) -> bool:
        """检查缓冲区是否有足够数据"""
        with self._lock:
            total = sum(c.shape[1] for c in self._buffer)
        return total >= n_frames

    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(c.shape[1] for c in self._buffer)
