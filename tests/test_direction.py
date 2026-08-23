"""
Phase 2 验证脚本 — 测试方向估算算法

运行:
    python tests/test_direction.py

测试项目:
    1. ILD 角度映射
    2. ITD 角度映射
    3. GCC-PHAT 对已知延迟信号的检测精度
    4. 信噪比影响
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.processor.gcc_phat import gcc_phat, compute_itd, compute_ild
from src.processor.direction import ild_to_azimuth, itd_to_azimuth, estimate_azimuth


# ============================================================
# 测试辅助
# ============================================================

def create_test_signal(fs=48000, duration=0.05, freq=500, delay_samples=0):
    """
    创建测试用的正弦脉冲信号。

    delay_samples > 0 → 右声道延迟 (模拟左先到)
    delay_samples < 0 → 左声道延迟 (模拟右先到)
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * freq * t) * np.exp(-t * 50)  # 衰减包络

    if delay_samples > 0:
        left = np.concatenate([signal, np.zeros(delay_samples)])
        right = np.concatenate([np.zeros(delay_samples), signal])
    elif delay_samples < 0:
        delay_samples = -delay_samples
        right = np.concatenate([signal, np.zeros(delay_samples)])
        left = np.concatenate([np.zeros(delay_samples), signal])
    else:
        left = signal.copy()
        right = signal.copy()

    return left, right


# ============================================================
# 测试用例
# ============================================================

def test_ild_mapping():
    """测试 ILD → 角度映射"""
    print("--- ILD 映射 ---")
    test_cases = [
        (0.0, 0.0, "正前方"),
        (15.0, 90.0, "极左"),
        (-15.0, -90.0, "极右"),
        (7.5, 45.0, "左前方45°"),
        (-7.5, -45.0, "右前方45°"),
    ]
    all_pass = True
    for ild, expected, desc in test_cases:
        result = ild_to_azimuth(ild)
        ok = abs(result - expected) < 1.0
        status = "✓" if ok else "✗"
        print(f"  {status} ILD={ild:+5.1f}dB → {result:+5.1f}° (期望 {expected:+5.1f}°) [{desc}]")
        if not ok:
            all_pass = False
    return all_pass


def test_itd_mapping():
    """测试 ITD → 角度映射"""
    print("\n--- ITD 映射 ---")
    max_tau = 0.00065
    test_cases = [
        (0.0, 0.0, "正前方"),
        (max_tau, 90.0, "极左"),
        (-max_tau, -90.0, "极右"),
        (max_tau * 0.7071, 45.0, "左前方45°"),
    ]
    all_pass = True
    for tau, expected, desc in test_cases:
        result = itd_to_azimuth(tau, max_tau=max_tau)
        ok = abs(result - expected) < 1.5
        status = "✓" if ok else "✗"
        print(f"  {status} tau={tau*1e6:+5.0f}μs → {result:+5.1f}° (期望 {expected:+5.1f}°) [{desc}]")
        if not ok:
            all_pass = False
    return all_pass


def test_gcc_phat_accuracy():
    """测试 GCC-PHAT 对已知延迟的检测精度"""
    print("\n--- GCC-PHAT 精度 ---")
    fs = 48000
    test_delays = [0, 2, 5, 10, 20, 31]  # 采样点延迟
    # 31 samples @ 48kHz = 0.65ms (max expected ITD)

    all_pass = True
    for delay in test_delays:
        left, right = create_test_signal(fs=fs, delay_samples=delay)
        tau, _ = gcc_phat(left, right, fs=fs, max_tau=0.001, interp=16)

        expected_tau = delay / fs
        error_us = abs(tau - expected_tau) * 1e6
        ok = error_us < 1.0  # 误差 < 1μs

        # 延迟=0 时用不同标准 (tau 可正可负)
        if delay == 0:
            ok = abs(tau) * 1e6 < 1.0

        status = "✓" if ok else "✗"
        print(f"  {status} 延迟 {delay:2d} samples → tau={tau*1e6:+7.2f}μs (期望 {expected_tau*1e6:+.2f}μs, 误差 {error_us:.2f}μs)")
        if not ok:
            all_pass = False

    return all_pass


def test_full_estimation():
    """测试完整方向估算流程"""
    print("\n--- 完整方向估算 ---")
    fs = 48000

    test_cases = [
        # delay_samples (正=左先到=声音靠左=正角度)
        (0, "正前方", 0.0),
        (10, "偏左", 30.0),
        (-10, "偏右", -30.0),
    ]

    all_pass = True
    for delay, desc, expected_sign in test_cases:
        left, right = create_test_signal(fs=fs, delay_samples=delay)
        azimuth = estimate_azimuth(left, right, fs=fs)

        correct_sign = (azimuth * expected_sign >= 0) if expected_sign != 0 else (abs(azimuth) < 20)
        status = "✓" if correct_sign else "✗"
        print(f"  {status} {desc}: azimuth = {azimuth:+5.1f}° (期望 {'左' if expected_sign > 0 else '右' if expected_sign < 0 else '中'})")
        if not correct_sign:
            all_pass = False

    return all_pass


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  GoodEar — Phase 2: 方向估算测试")
    print("=" * 60)
    print()

    results = {
        "ILD 映射": test_ild_mapping(),
        "ITD 映射": test_itd_mapping(),
        "GCC-PHAT 精度": test_gcc_phat_accuracy(),
        "完整方向估算": test_full_estimation(),
    }

    print("\n" + "=" * 60)
    print("  测试结果汇总:")
    for name, ok in results.items():
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"    {status}: {name}")

    all_pass = all(results.values())
    print()
    if all_pass:
        print("  ★ 全部测试通过！方向估算核心算法工作正常。")
    else:
        print("  ! 部分测试失败，请检查算法实现。")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
