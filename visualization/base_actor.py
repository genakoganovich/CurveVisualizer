#visualization/base_actor.py
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
        return smoothed

    def update(self, plotter, t: float):
        """
        Обновить актор БЕЗ удаления (не мигает)
        """
        position, direction = self._compute_geometry(t)

        position = self._smooth_value(position, self._last_position, is_vector=False)
        direction = self._smooth_value(direction, self._last_direction, is_vector=False)

        self._last_position = position.copy() if isinstance(position, np.ndarray) else position
        self._last_direction = direction.copy() if isinstance(direction, np.ndarray) else direction

        if self._actor is None:
            self._actor = self._create_mesh(position, direction, plotter)
        else:
            self._update_actor_position(position, direction, plotter)

    def _update_actor_position(self, position: np.ndarray, direction: np.ndarray, plotter):
        """
        ★ Обновляет позицию и направление актора
        """
        try:
            new_mesh = self._create_mesh_geometry(position, direction)

            if new_mesh is not None and self._actor is not None:
                # ★ Правильный способ для PyVista
                try:
                    # Получаем mapper и его input (текущий меш)
                    mapper = self._actor.GetMapper()
                    if mapper:
                        # ★ Используем SetInputData для быстрого обновления
                        mapper.SetInputData(new_mesh)
                        # ★ Уведомляем mapper что данные изменились
                        self._actor.Modified()
                except AttributeError:
                    # Fallback: пересоздаем если что-то пошло не так
                    try:
                        plotter.remove_actor(self._actor)
                    except:
                        pass
                    self._actor = self._create_mesh(position, direction, plotter)
        except Exception as e:
            print(f"⚠️ Ошибка обновления актора: {e}")

    def _create_mesh_geometry(self, position: np.ndarray, direction: np.ndarray):
        """
        Создать объект mesh БЕЗ добавления в plotter
        Переопределяется в подклассах
        """
        return None

    @abstractmethod
    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """Создать и добавить mesh в plotter (первый раз)"""
        pass