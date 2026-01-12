# visualization/actors.py

import pyvista as pv
import numpy as np


class ArrowActor:
    """Стрелка с плавным движением"""

    def __init__(self, curve, arrow_type: str = "tangent", scale: float = 0.3,
                 color: str = "white", smoothing: float = 0.7):
        """
        Args:
            smoothing: коэффициент сглаживания (0-1)
                      0 = обновлять каждый кадр
                      1 = максимальное сглаживание
        """
        self.curve = curve
        self.arrow_type = arrow_type
        self.scale = scale
        self.color = color
        self.smoothing = smoothing  # ★ Новое

        self._direction_func = self._get_direction_func()
        self._actor = None
        self._last_position = None
        self._last_direction = None

    def _get_direction_func(self):
        methods = {
            "tangent": self.curve.tangent,
            "normal": self.curve.normal,
            "binormal": self.curve.binormal,
        }
        method = methods.get(self.arrow_type)
        if method is None:
            raise ValueError(f"Unknown arrow_type: {self.arrow_type}")
        return method

    def _get_direction(self, t: float) -> np.ndarray:
        t_arr = np.array([t])
        return self._direction_func(t_arr)[0]

    def update(self, plotter, t: float):
        """Обновить позицию с сглаживанием"""
        position = self.curve.position(np.array([t]))[0]
        direction = self._get_direction(t)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # ★ Первый раз - просто сохраняем
        if self._last_position is None:
            self._last_position = position
            self._last_direction = direction
        else:
            # ★ Интерполируем между старым и новым значением
            position = self._last_position * self.smoothing + position * (1 - self.smoothing)
            direction = self._last_direction * self.smoothing + direction * (1 - self.smoothing)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

        self._last_position = position
        self._last_direction = direction

        if self._actor is None:
            arrow = pv.Arrow(
                start=position,
                direction=direction,
                scale=self.scale
            )
            self._actor = plotter.add_mesh(arrow, color=self.color)
        else:
            try:
                plotter.remove_actor(self._actor)
            except:
                pass

            arrow = pv.Arrow(
                start=position,
                direction=direction,
                scale=self.scale
            )
            self._actor = plotter.add_mesh(arrow, color=self.color)