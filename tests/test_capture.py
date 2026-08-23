"""
Phase 1 验证脚本 — 测试 WASAPI loopback 音频捕获

运行:
    cd GoodEar
    python tests/test_capture.py

检查点:
    1. 能发现 WASAPI loopback 设备
    2. 能捕获立体声音频
    3. 左右声道数据可分离
    4. 实时显示左右声道 RMS 电平
"""

import sys
import os
import time
import signal

# 将 src 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.audio.capture import AudioCapture


def print_level_meter(left_level, right_level, width=40):
    """打印左右声道电平条"""
    left_bar = int(min(left_level, 1.0) * width)
    right_bar = int(min(right_level, 1.0) * width)

    left_str = '#' * left_bar + '-' * (width - left_bar)
    right_str = '#' * right_bar + '-' * (width - right_bar)

    # 使用控制字符实现原地刷新
    print(f'\rL [{left_str}] {left_level:.3f}  |  R [{right_str}] {right_level:.3f}', end='', flush=True)


def main():
    print('=' * 60)
    print('  GoodEar — Phase 1: 音频捕获测试')
    print('=' * 60)
    print()
    print('正在初始化 WASAPI loopback 捕获...')

    cap = AudioCapture()

    try:
        device = cap.start()
        print(f'✓ 设备已连接: {device["name"]}')
        print(f'  采样率: {device["sample_rate"]} Hz')
        print(f'  声道: {device["channels"]}')
        print()
        print('正在捕获音频... 按 Ctrl+C 退出')
        print('(播放任何音频来测试 — 比如打开一个 YouTube 视频)')
        print()
        print('L [左声道]                                   R [右声道]')
        print('-' * 90)

        count = 0
        while True:
            # 获取最新数据
            data = cap.get_latest()
            if data is None:
                time.sleep(0.01)
                continue

            left = data[0, :]   # 左声道
            right = data[1, :]  # 右声道

            # 计算 RMS
            rms_l = float(np.sqrt(np.mean(left ** 2)) + 1e-9)
            rms_r = float(np.sqrt(np.mean(right ** 2)) + 1e-9)

            print_level_meter(rms_l * 10, rms_r * 10)  # 放大 10 倍显示

            count += 1

    except KeyboardInterrupt:
        print('\n\n用户中断。')
    except Exception as e:
        print(f'\n错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        cap.stop()
        print('捕获已停止。')


if __name__ == '__main__':
    main()
