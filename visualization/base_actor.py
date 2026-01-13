import numpy as np
import pyvista as pv
from abc import ABC, abstractmethod


class BaseActor(ABC):
    """Базовый класс для всех акторов с встроенным сглаживанием"""

    arrow_type = "base"

    def __init__(self, curve, color: str = "white", smoothing: float = 0.0):
        """
        Args:
            curve: объект Curve3D
            color: цвет актора
            smoothing: коэффициент сглаживания (0-1)
                      0 = без сглаживания
                      1 = максимальное сглаживание
        """
        self.curve = curve
        self.color = color
        self.smoothing = smoothing

        self._actor = None
        self._last_position = None
        self._last_direction = None

    @abstractmethod
    def _compute_geometry(self, t: float) -> tuple:
        """
        Вычислить геометрию актора

        Returns:
            (position, direction) или (position, shape)
        """
        pass

    def _smooth_value(self, new_value, last_value, is_vector: bool = True):
        """
        Сгладить значение между старым и новым

        Args:
            new_value: новое значение (float или np.ndarray)
            last_value: последнее значение
            is_vector: является ли векторной величиной

        Returns:
            сглаженное значение
        """
        if last_value is None:
            return new_value

        smoothed = last_value * self.smoothing + new_value * (1 - self.smoothing)

        # Если это вектор, нормализуем
        if is_vector and isinstance(smoothed, np.ndarray) and smoothed.ndim == 1:
            norm = np.linalg.norm(smoothed)
            if norm > 1e-10:
                # Сохраняем длину оригинального вектора
                smoothed = smoothed / norm * np.linalg.norm(new_value)

        return smoothed

    def update(self, plotter, t: float):
        """Обновить актор с автоматическим сглаживанием"""
        # Вычисляем геометрию
        position, direction = self._compute_geometry(t)

        # ★ Применяем сглаживание
        position = self._smooth_value(position, self._last_position)
        direction = self._smooth_value(direction, self._last_direction, is_vector=True)

        # Сохраняем для следующей итерации
        self._last_position = position
        self._last_direction = direction

        # Удаляем старый актор
        if self._actor is not None:
            try:
                plotter.remove_actor(self._actor)
            except:
                pass

        # Создаем и добавляем новый актор
        self._actor = self._create_mesh(position, direction, plotter)

    @abstractmethod
    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """Создать и добавить mesh в plotter"""
        pass