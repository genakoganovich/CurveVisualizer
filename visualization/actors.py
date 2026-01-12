# visualization/actors.py

import pyvista as pv
import numpy as np


class ArrowActor:
    """Стрелка на кривой (оптимизированная версия)"""

    def __init__(self, curve, arrow_type: str = "tangent", scale: float = 0.3, color: str = "white"):
        self.curve = curve
        self.arrow_type = arrow_type
        self.scale = scale
        self.color = color

        self._direction_func = self._get_direction_func()
        self._actor = None

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
        """Обновить позицию стрелки"""
        position = self.curve.position(np.array([t]))[0]
        direction = self._get_direction(t)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # ★ ПЕРВЫЙ РАЗ: создаем стрелку
        if self._actor is None:
            # Используем pv.Arrow для красивой стрелки
            arrow = pv.Arrow(
                start=position,
                direction=direction,
                scale=self.scale
            )
            self._actor = plotter.add_mesh(arrow, color=self.color)
        else:
            # ★ Удаляем и заново добавляем (но это быстро для Arrow)
            plotter.remove_actor(self._actor)

            arrow = pv.Arrow(
                start=position,
                direction=direction,
                scale=self.scale
            )
            self._actor = plotter.add_mesh(arrow, color=self.color)