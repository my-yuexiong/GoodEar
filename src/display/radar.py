"""
pygame 实时雷达显示

极坐标雷达，显示检测到的声音事件方向和距离。
支持扫描线动画、检测点衰减拖尾、HUD 信息叠加。

用独立线程运行，与音频处理管线解耦。
"""

import math
import time
import threading

import numpy as np

from src.config import (
    RADAR_SIZE,
    RADAR_FPS,
    SWEEP_SPEED,
    POINT_LIFETIME,
    TRAIL_FADE_SPEED,
    MAX_DIST_M,
    BG_COLOR,
    GRID_COLOR,
    SWEEP_COLOR,
    POINT_COLOR,
    TEXT_COLOR,
)


class RadarDisplay:
    """
    pygame 极坐标雷达显示器。

    雷达布局:
        - 正前方 = 顶部 (0° = 上)
        - 左侧 = 左侧 (-90° = 左)
        - 右侧 = 右侧 (+90° = 右)
        - 中心 = 玩家位置
        - 半径 = 距离

    用法:
        radar = RadarDisplay()
        radar.start()
        radar.add_detection(azimuth_deg, distance_m)
        radar.stop()
    """

    def __init__(
        self,
        size: int = RADAR_SIZE,
        fps: int = RADAR_FPS,
        sweep_speed: float = SWEEP_SPEED,
        point_lifetime: float = POINT_LIFETIME,
        max_distance: float = MAX_DIST_M,
    ):
        self.size = size
        self.fps = fps
        self.sweep_speed = sweep_speed
        self.point_lifetime = point_lifetime
        self.max_distance = max_distance

        self.center_x = size // 2
        self.center_y = size // 2
        self.radius = size // 2 - 40  # 留边距给文字

        # 扫描线角度 (度)
        self.sweep_angle = 0.0

        # 检测点列表
        # 每个点: {'azimuth', 'distance', 'timestamp', 'alpha'}
        self._points: list[dict] = []
        self._points_lock = threading.Lock()

        # 状态
        self._running = False
        self._thread: threading.Thread | None = None

        # pygame 对象 (在 _run 中初始化)
        self._trail_surface = None
        self._font = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """在独立线程启动雷达显示。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """停止雷达显示。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def add_detection(self, azimuth_deg: float, distance_m: float):
        """
        添加一个检测事件。

        Args:
            azimuth_deg: 方位角 (度)，0=正前方，正=左
            distance_m: 估算距离 (米)
        """
        now = time.time()
        with self._points_lock:
            self._points.append({
                'azimuth': azimuth_deg,
                'distance': distance_m,
                'timestamp': now,
            })
            # 限制最大点数 (避免内存增长)
            if len(self._points) > 100:
                self._points = self._points[-100:]

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _run(self):
        """主渲染循环 (运行在独立线程)。"""
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((self.size, self.size))
        pygame.display.set_caption("GoodEar - Audio Radar")
        clock = pygame.time.Clock()
        self._font = pygame.font.SysFont('consolas', 12)

        # 拖尾表面 (带 alpha 通道)
        self._trail_surface = pygame.Surface(
            (self.size, self.size), pygame.SRCALPHA
        )

        last_time = time.time()

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break

            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # 更新扫描线
            self.sweep_angle = (self.sweep_angle + self.sweep_speed * dt) % 360

            # 清理过期的检测点
            now = time.time()
            with self._points_lock:
                self._points = [
                    p for p in self._points
                    if now - p['timestamp'] < self.point_lifetime
                ]

            # 渲染
            screen.fill(BG_COLOR)
            self._draw_grid(screen)
            self._draw_sweep_trail()
            screen.blit(self._trail_surface, (0, 0))
            self._draw_sweep_line(screen)
            self._draw_points(screen, now)
            self._draw_center(screen)
            self._draw_info(screen, dt)

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def _polar_to_screen(self, azimuth_deg: float, distance_m: float) -> tuple[int, int]:
        """
        极坐标 → 屏幕坐标。

        方位角: 0=正前方(上), 正=左侧, 负=右侧
        距离: 米 → 像素
        """
        # 转换为屏幕角度 (0=右, 逆时针)
        screen_angle = 90.0 - azimuth_deg
        rad = math.radians(screen_angle)

        # 距离映射到半径
        r = (distance_m / self.max_distance) * self.radius
        r = min(r, self.radius)

        x = self.center_x + r * math.cos(rad)
        y = self.center_y - r * math.sin(rad)
        return (int(x), int(y))

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, screen):
        """绘制同心圆和十字线。"""
        for pct in [0.25, 0.50, 0.75, 1.0]:
            r = int(self.radius * pct)
            import pygame
            pygame.draw.circle(screen, GRID_COLOR, (self.center_x, self.center_y), r, 1)

            # 距离标签
            if pct > 0:
                dist = pct * self.max_distance
                label = self._font.render(f'{dist:.0f}m', True, GRID_COLOR)
                screen.blit(label, (self.center_x + 4, self.center_y - r + 2))

        # 十字线
        import pygame
        pygame.draw.line(screen, GRID_COLOR,
                         (self.center_x, self.center_y - self.radius),
                         (self.center_x, self.center_y + self.radius), 1)
        pygame.draw.line(screen, GRID_COLOR,
                         (self.center_x - self.radius, self.center_y),
                         (self.center_x + self.radius, self.center_y), 1)

    def _draw_sweep_line(self, screen):
        """绘制当前扫描线。"""
        sweep_rad = math.radians(90.0 - self.sweep_angle)
        end_x = self.center_x + self.radius * math.cos(sweep_rad)
        end_y = self.center_y - self.radius * math.sin(sweep_rad)
        import pygame
        pygame.draw.line(screen, SWEEP_COLOR,
                         (self.center_x, self.center_y),
                         (int(end_x), int(end_y)), 2)

    def _draw_sweep_trail(self):
        """绘制扫描线拖尾 (扇形渐隐)。"""
        import pygame

        # 整体衰减拖尾表面
        fade_array = pygame.surfarray.pixels_alpha(self._trail_surface)
        fade_array[:] = np.maximum(
            fade_array.astype(np.int16) - TRAIL_FADE_SPEED, 0
        )
        del fade_array  # 释放 surface lock

        # 在新位置绘制扇形尾迹
        for offset in range(1, 36):
            angle = self.sweep_angle - offset * 2
            alpha = max(255 - offset * 7, 0)
            rad = math.radians(90.0 - angle)
            ex = self.center_x + self.radius * math.cos(rad)
            ey = self.center_y - self.radius * math.sin(rad)
            color = (*SWEEP_COLOR, alpha)
            pygame.draw.line(self._trail_surface, color,
                             (self.center_x, self.center_y),
                             (int(ex), int(ey)), 1)

    def _draw_points(self, screen, now):
        """绘制检测点 (带年龄衰减)。"""
        import pygame

        with self._points_lock:
            points = list(self._points)

        for p in points:
            age = now - p['timestamp']
            alpha = max(int(255 * (1.0 - age / self.point_lifetime)), 30)
            pos = self._polar_to_screen(p['azimuth'], p['distance'])

            # 绘制带 alpha 的圆点
            s = pygame.Surface((14, 14), pygame.SRCALPHA)
            # 外圈 (发光效果)
            pygame.draw.circle(s, (*POINT_COLOR, alpha // 3), (7, 7), 7)
            # 实心
            pygame.draw.circle(s, (*POINT_COLOR, alpha), (7, 7), 4)
            screen.blit(s, (pos[0] - 7, pos[1] - 7))

    def _draw_center(self, screen):
        """绘制中心点 (玩家位置)。"""
        import pygame
        pygame.draw.circle(screen, TEXT_COLOR,
                           (self.center_x, self.center_y), 4, 1)
        pygame.draw.circle(screen, TEXT_COLOR,
                           (self.center_x, self.center_y), 2)

    def _draw_info(self, screen, dt):
        """绘制 HUD 信息。"""
        import pygame
        fps = 1.0 / max(dt, 0.001)
        texts = [
            f"FPS: {fps:.0f}",
            f"Points: {len(self._points)}",
            f"Angle: {self.sweep_angle:.0f}",
        ]
        for i, text in enumerate(texts):
            surf = self._font.render(text, True, TEXT_COLOR)
            screen.blit(surf, (8, 8 + i * 16))
