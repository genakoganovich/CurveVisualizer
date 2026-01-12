import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple


class Curve3D:
    """Параметрическая 3D кривая на основе CubicSpline"""

    def __init__(self, points: np.ndarray):
        """
        Args:
            points: (N, 3) массив контрольных точек
        """
        self.points = points
        self.t_param = np.linspace(0, 1, len(points))

        # Создаем сплайны для x, y, z
        self.spline_x = CubicSpline(self.t_param, points[:, 0], bc_type='not-a-knot')
        self.spline_y = CubicSpline(self.t_param, points[:, 1], bc_type='not-a-knot')
        self.spline_z = CubicSpline(self.t_param, points[:, 2], bc_type='not-a-knot')

        # Предварительно вычисляем длину кривой
        self._precompute_arc_length()

    def _precompute_arc_length(self, num_samples=1000):
        """Предварительно вычисляем накопленную длину"""
        t_samples = np.linspace(0, 1, num_samples)
        positions = self.position(t_samples)

        # Дифференциальные отрезки длины
        diffs = np.diff(positions, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        self.cum_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
        self.total_length = self.cum_lengths[-1]
        self.t_samples = t_samples

    # ============= ПОЗИЦИЯ И ПРОИЗВОДНЫЕ =============

    def position(self, t: np.ndarray) -> np.ndarray:
        """Позиция P(t)"""
        return np.column_stack([
            self.spline_x(t),
            self.spline_y(t),
            self.spline_z(t)
        ])

    def velocity(self, t: np.ndarray) -> np.ndarray:
        """Первая производная dP/dt"""
        return np.column_stack([
            self.spline_x(t, 1),
            self.spline_y(t, 1),
            self.spline_z(t, 1)
        ])

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        """Вторая производная d²P/dt²"""
        return np.column_stack([
            self.spline_x(t, 2),
            self.spline_y(t, 2),
            self.spline_z(t, 2)
        ])

    def jerk(self, t: np.ndarray) -> np.ndarray:
        """Третья производная d³P/dt³"""
        return np.column_stack([
            self.spline_x(t, 3),
            self.spline_y(t, 3),
            self.spline_z(t, 3)
        ])

    # ============= ГЕОМЕТРИЯ (FRENET FRAME) =============

    def tangent(self, t: np.ndarray) -> np.ndarray:
        """Касательный вектор (нормализованный)"""
        vel = self.velocity(t)
        norms = np.linalg.norm(vel, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1  # избегаем деления на 0
        return vel / norms

    def curvature(self, t: np.ndarray) -> np.ndarray:
        """Кривизна κ(t)"""
        vel = self.velocity(t)
        acc = self.acceleration(t)

        cross = np.cross(vel, acc)
        cross_norm = np.linalg.norm(cross, axis=1)
        vel_norm = np.linalg.norm(vel, axis=1)

        # κ = ||dP/dt × d²P/dt²|| / ||dP/dt||³
        curvature = cross_norm / (vel_norm ** 3 + 1e-10)
        return curvature

    def radius_of_curvature(self, t: np.ndarray) -> np.ndarray:
        """Радиус кривизны R = 1/κ"""
        kappa = self.curvature(t)
        return np.where(kappa > 1e-10, 1.0 / kappa, np.inf)

    def frenet_frame(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Возвращает (T, N, B) — Frenet-Serret frame
        T: касательная
        N: нормаль (главная)
        B: бинормаль
        """
        vel = self.velocity(t)
        acc = self.acceleration(t)

        # Касательная T
        T = self.tangent(t)

        # dT/ds = κ * N
        # dT/dt = (dT/ds) * (ds/dt) = κ * N * ||dP/dt||
        # => N = (dT/dt) / (κ * ||dP/dt||)

        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)

        # dT/dt численно
        dt = 1e-6
        T_plus = self.tangent(np.clip(t + dt, 0, 1))
        dT_dt = (T_plus - T) / dt

        N = dT_dt / (np.linalg.norm(dT_dt, axis=1, keepdims=True) + 1e-10)

        # Бинормаль B = T × N
        B = np.cross(T, N)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)

        return T, N, B

    def torsion(self, t: np.ndarray) -> np.ndarray:
        """Кручение τ(t)"""
        vel = self.velocity(t)
        acc = self.acceleration(t)
        jer = self.jerk(t)

        # τ = (dP/dt × d²P/dt²) · d³P/dt³ / ||dP/dt × d²P/dt²||²
        cross = np.cross(vel, acc)
        cross_norm_sq = np.sum(cross ** 2, axis=1)

        torsion = np.sum(cross * jer, axis=1) / (cross_norm_sq + 1e-10)
        return torsion

    # ============= УДОБНЫЕ МЕТОДЫ ДЛЯ ДОСТУПА К FRENET FRAME =============

    def normal(self, t: np.ndarray) -> np.ndarray:
        """Главная нормаль N(t)"""
        _, N, _ = self.frenet_frame(t)
        return N

    def binormal(self, t: np.ndarray) -> np.ndarray:
        """Бинормаль B(t)"""
        _, _, B = self.frenet_frame(t)
        return B

    def tangent_vector(self, t: np.ndarray) -> np.ndarray:
        """Касательный вектор T(t) через frenet_frame"""
        T, _, _ = self.frenet_frame(t)
        return T

    # ============= КИНЕМАТИКА =============

    def speed(self, t: np.ndarray) -> np.ndarray:
        """Скорость |dP/dt|"""
        vel = self.velocity(t)
        return np.linalg.norm(vel, axis=1)

    def arc_length(self, t_start: float = 0, t_end: float = 1) -> float:
        """Длина дуги от t_start до t_end"""
        # Использует предварительно вычисленную длину
        idx_start = np.searchsorted(self.t_samples, t_start)
        idx_end = np.searchsorted(self.t_samples, t_end)
        return self.cum_lengths[idx_end] - self.cum_lengths[idx_start]

    def angular_velocity(self, t: np.ndarray) -> np.ndarray:
        """Угловая скорость ω = (dP/dt × d²P/dt²) / ||dP/dt||²"""
        vel = self.velocity(t)
        acc = self.acceleration(t)

        cross = np.cross(vel, acc)
        vel_norm_sq = np.sum(vel ** 2, axis=1, keepdims=True)

        return cross / (vel_norm_sq + 1e-10)

    def tangential_acceleration(self, t: np.ndarray) -> np.ndarray:
        """Тангенциальное ускорение a_t = (dP/dt · d²P/dt²) / ||dP/dt||"""
        vel = self.velocity(t)
        acc = self.acceleration(t)

        vel_norm = np.linalg.norm(vel, axis=1)
        a_t = np.sum(vel * acc, axis=1) / (vel_norm + 1e-10)

        return a_t

    def normal_acceleration(self, t: np.ndarray) -> np.ndarray:
        """Нормальное ускорение a_n = ||dP/dt × d²P/dt²|| / ||dP/dt||²"""
        vel = self.velocity(t)
        acc = self.acceleration(t)

        cross = np.cross(vel, acc)
        cross_norm = np.linalg.norm(cross, axis=1)
        vel_norm = np.linalg.norm(vel, axis=1)

        a_n = cross_norm / (vel_norm ** 2 + 1e-10)

        return a_n