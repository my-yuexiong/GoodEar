"""
WASAPI Loopback 音频捕获
通过 PyAudioWPatch 将系统音频输出捕获为立体声 numpy 数组。

依赖:
    pip install PyAudioWPatch
"""

import numpy as np

from src.config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, FORMAT_BITS, BUFFER_CHUNKS
from src.audio.buffer import RingBuffer


class AudioCapture:
    """
    WASAPI loopback 实时立体声音频捕获器。

    使用 callback 模式：PyAudio 在高优先级线程中回调，
    callback 仅做缓冲，所有重活由外部处理线程完成。

    用法:
        cap = AudioCapture()
        cap.start()
        # 在另一个线程/循环中:
        data = cap.get_latest()
        cap.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_size: int = CHUNK_SIZE,
        buffer_chunks: int = BUFFER_CHUNKS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self._buffer = RingBuffer(max_chunks=buffer_chunks)
        self._running = False

        self._pyaudio = None
        self._pa_module = None   # 保存 pyaudio 模块引用，供 callback 使用
        self._stream = None
        self._device_info = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> dict:
        """
        启动 WASAPI loopback 捕获。

        Returns:
            dict: 设备信息 {'name', 'sample_rate', 'channels'}

        Raises:
            RuntimeError: WASAPI 不可用或未找到 loopback 设备
        """
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise ImportError(
                "请安装 PyAudioWPatch: pip install PyAudioWPatch\n"
                "如果安装失败，可尝试替代方案: pip install sounddevice"
            )

        self._pa_module = pyaudio  # 保存模块引用，供 callback 闭包使用
        self._pyaudio = pyaudio.PyAudio()

        # 查找 WASAPI host API
        try:
            wasapi_info = self._pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            self._cleanup()
            raise RuntimeError(
                "WASAPI 不可用。请确保:\n"
                "  1. 操作系统为 Windows Vista 或更高版本\n"
                "  2. 音频服务正在运行"
            )

        # 获取默认扬声器
        default_speakers = self._pyaudio.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        # 查找对应的 loopback 设备
        if not default_speakers["isLoopbackDevice"]:
            found = False
            for loopback in self._pyaudio.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    found = True
                    break
            if not found:
                self._cleanup()
                raise RuntimeError(
                    "未找到 WASAPI loopback 设备。\n"
                    "请确认默认播放设备已启用。"
                )

        self._device_info = {
            "name": default_speakers["name"],
            "sample_rate": int(default_speakers["defaultSampleRate"]),
            "channels": self.channels,
        }
        self.sample_rate = self._device_info["sample_rate"]

        print(f"[AudioCapture] 设备: {self._device_info['name']}")
        print(f"[AudioCapture] 采样率: {self._device_info['sample_rate']} Hz")
        print(f"[AudioCapture] chunk: {self.chunk_size} 帧 (~{self.chunk_size / self.sample_rate * 1000:.1f}ms)")

        # 打开 loopback 流 (callback 模式)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            input_device_index=default_speakers["index"],
            stream_callback=self._audio_callback,
        )
        self._running = True
        self._stream.start_stream()

        return self._device_info

    def stop(self):
        """停止捕获并释放资源。"""
        self._running = False

        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self._cleanup()

    def get_latest(self, n_frames: int | None = None) -> np.ndarray | None:
        """
        获取缓冲区中的最新音频数据。

        Args:
            n_frames: 返回最近 N 帧，None = 全部可用

        Returns:
            (2, N) float32 [-1.0, 1.0]，或 None
        """
        return self._buffer.get_latest(n_frames)

    def ready_for(self, n_frames: int) -> bool:
        """是否有足够的缓冲数据。"""
        return self._buffer.ready_for(n_frames)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_info(self) -> dict | None:
        return self._device_info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio 回调 — 在高优先级线程中运行，只做轻量操作。

        将 int16 bytes 解码为 float32 numpy 数组并推入环形缓冲区。
        """
        if status:
            # status 标志可能包含 inputUnderflow 等警告
            pass

        # bytes → float32 stereo (2, frame_count)
        raw = np.frombuffer(in_data, dtype=np.int16)
        stereo = raw.reshape(-1, 2).T.astype(np.float32) / 32768.0

        self._buffer.append(stereo)

        return (in_data, self._pa_module.paContinue)

    def _cleanup(self):
        """释放 PyAudio 资源。"""
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
