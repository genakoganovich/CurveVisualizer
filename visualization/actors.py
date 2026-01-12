# visualization/actors.py

import numpy as np
import pyvista as pv


class ArrowActor:
    """Актор - стрелка на кривой"""
    _DIRECTION_METHODS = {
        "tangent": "tangent",
        "normal": "normal",
        "binormal": "binormal",
    }

    def __init__(self, curve, arrow_type: str = "tangent", scale: float = 0.3, color: str = "red"):
        """
        Args:
            curve: объект кривой
            arrow_type: тип стрелки ("tangent", "normal", "binormal")
            scale: масштаб стрелки
            color: цвет стрелки
        """
        self.curve = curve
        self.arrow_type = arrow_type
        self.scale = scale
        self.color = color
        self._direction_func = self._get_direction_func()

    def _get_direction_func(self):
        """Получить функцию один раз при инициализации"""
        method_name = self._DIRECTION_METHODS.get(self.arrow_type)
        if method_name is None:
            raise ValueError(f"Unknown arrow_type: {self.arrow_type}")
        return getattr(self.curve, method_name)

    def _get_direction(self, t: float) -> np.ndarray:
        """Получить направление в зависимости от типа"""
        t_arr = np.array([t])
        return self._direction_func(t_arr)[0]

    def update(self, plotter: pv.Plotter, t: float):
        """Обновить стрелку"""
        t_arr = np.array([t])
        pos = self.curve.position(t_arr)[0]
        direction = self._get_direction(t)

        # Удаляем старую стрелку (но не кривую)
        actors_list = list(plotter.actors.values())
        for actor in actors_list[1:]:
            try:
                plotter.remove_actor(actor, reset_camera=False)
            except:
                pass

        # Добавляем новую стрелку
        end_pos = pos + direction * self.scale
        arrow = pv.Line(pos, end_pos)
        plotter.add_mesh(arrow, color=self.color, line_width=4)