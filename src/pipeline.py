"""
主处理 Pipeline — 串联音频捕获、检测、方向估算、显示

线程架构:
    1. 音频 callback 线程 → 环形缓冲区 (PyAudio 管理)
    2. 主处理线程 → 从缓冲区取数据、检测、估算
    3. 雷达 UI 线程 → pygame 渲染循环
"""

import time
import threading

import numpy as np

from src.config import PROCESS_WINDOW, FFT_HOP_SIZE, FFT_FRAME_SIZE
from src.audio.capture import AudioCapture
from src.processor.footstep import FootstepDetector
from src.processor.direction import estimate_azimuth
from src.processor.distance import estimate_distance, distance_to_zone
from src.display.radar import RadarDisplay


class Pipeline:
    """
    音频雷达主管线。

    用法:
        pipeline = Pipeline()
        pipeline.start()
        # 阻塞运行直到 Ctrl+C
        pipeline.stop()
    """

    def __init__(self):
        self.capture = AudioCapture()
        self.detector = FootstepDetector()
        self.radar = RadarDisplay()
        self._running = False

        # 统计
        self._total_detections = 0
        self._footstep_count = 0
        self._stats_lock = threading.Lock()

    def start(self):
        """启动整个 pipeline。"""
        print("=" * 60)
        print("  GoodEar - Game Audio Radar")
        print("=" * 60)
        print()
        print("[Pipeline] 启动音频捕获...")
        device = self.capture.start()

        print("[Pipeline] 启动雷达显示...")
        self.radar.start()

        self._running = True
        print("[Pipeline] 全管线运行中，按 Ctrl+C 退出")
        print()

        try:
            while self._running:
                # 等待足够的数据
                if not self.capture.ready_for(PROCESS_WINDOW):
                    time.sleep(0.005)
                    continue

                # 获取处理窗口
                stereo = self.capture.get_latest(PROCESS_WINDOW)
                if stereo is None:
                    time.sleep(0.005)
                    continue

                # 脚步声检测
                events = self.detector.process(stereo)

                for event in events:
                    if event['sound_type'] != 'footstep':
                        continue  # 只显示脚步声

                    # 方位角在 detection 中已经估算
                    azimuth = event['azimuth']

                    # 距离估算
                    distance = estimate_distance(event['amplitude_db'])

                    # 发送到雷达
                    self.radar.add_detection(azimuth, distance)

                    # 统计
                    with self._stats_lock:
                        self._total_detections += 1
                        self._footstep_count += 1

                    # 控制台输出 (每 10 次打印一次，避免刷屏)
                    if self._footstep_count % 10 == 0:
                        zone = distance_to_zone(distance)
                        print(
                            f"\r[检测] 方位: {azimuth:+5.1f}°  "
                            f"距离: {distance:4.1f}m ({zone})  "
                            f"总数: {self._total_detections}    ",
                            end='', flush=True,
                        )

        except KeyboardInterrupt:
            print("\n[Pipeline] 收到中断信号")

        finally:
            self.stop()

    def stop(self):
        """停止 pipeline。"""
        self._running = False
        print("[Pipeline] 停止音频捕获...")
        self.capture.stop()
        print("[Pipeline] 停止雷达显示...")
        self.radar.stop()
        print(f"[Pipeline] 已停止。共检测到 {self._total_detections} 个脚步事件")
