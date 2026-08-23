"""
GoodEar — 游戏音频雷达

通过 WASAPI loopback 捕获系统音频，分析 FPS 游戏中的脚步声方位，
并在极坐标雷达上实时显示。

用法:
    python -m src.main

前提条件:
    1. 游戏音频设为"立体声耳机"模式
    2. 关闭 Windows 音频增强 (响度均衡、虚拟环绕)
"""

import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Pipeline


def main():
    print()
    print("   ____                 _  ______")
    print("  / ___| ___   ___   __| ||  ____| ___  __ _ _ __")
    print(" | |  _ / _ \\ / _ \\ / _` || |__   / _ \\/ _` | '__|")
    print(" | |_| | (_) | (_) | (_| ||  __| |  __/ (_| | |")
    print("  \\____|\\___/ \\___/ \\__,_||_|     \\___|\\__,_|_|")
    print()
    print("  Game Audio Radar — 实时脚步声方位雷达")
    print()

    pipeline = Pipeline()

    try:
        pipeline.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
