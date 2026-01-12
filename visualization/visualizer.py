import pyvista as pv
import numpy as np
from typing import Optional


class CurveVisualizer:
    """Визуализатор 3D кривых с PyVista"""

    def __init__(self, curve, num_samples: int = 500):
        """
        Args:
            curve: Curve3D объект
            num_samples: количество точек для визуализации
        """
        self.curve = curve
        self.num_samples = num_samples
        self.t = np.linspace(0, 1, num_samples)

        # Исправлено: убрали multi_samples
        self.plotter = pv.Plotter(shape=(2, 2),
                                  window_size=(1400, 1000))

        self._setup_viewports()

    def _setup_viewports(self):
        """Настроить 4 окна визуализации"""
        self.ax_geom = (0, 0)  # Геометрия
        self.ax_curv = (0, 1)  # Кривизна
        self.ax_kin = (1, 0)  # Кинематика
        self.ax_tor = (1, 1)  # Кручение

    def add_trajectory(self, color='blue', linewidth=2,
                       label='Trajectory', loc=(0, 0)):
        """Добавить траекторию"""
        positions = self.curve.position(self.t)

        # Создаем polyline
        poly = pv.Spline(positions)

        self.plotter.subplot(*loc)
        self.plotter.add_mesh(poly, color=color, line_width=linewidth,
                              label=label)
        self.plotter.add_legend()

    def add_tangent_vectors(self, step: int = 20, scale: float = 0.1,
                            color='red', loc=(0, 0)):
        """Добавить касательные векторы"""
        t_skip = self.t[::step]
        positions = self.curve.position(t_skip)
        tangents = self.curve.tangent(t_skip)

        self.plotter.subplot(*loc)
        for i, (pos, tan) in enumerate(zip(positions, tangents)):
            end = pos + tan * scale
            self.plotter.add_mesh(pv.Line(pos, end),
                                  color=color, line_width=2, opacity=0.7)

    def add_normal_vectors(self, step: int = 20, scale: float = 0.1,
                           loc=(0, 0)):
        """Добавить нормали"""
        t_skip = self.t[::step]
        positions = self.curve.position(t_skip)
        T, N, B = self.curve.frenet_frame(t_skip)

        self.plotter.subplot(*loc)
        for i, (pos, norm) in enumerate(zip(positions, N)):
            end = pos + norm * scale
            self.plotter.add_mesh(pv.Line(pos, end), color='green',
                                  line_width=2, opacity=0.7)

    def add_binormal_vectors(self, step: int = 20, scale: float = 0.1,
                             loc=(0, 0)):
        """Добавить бинормали"""
        t_skip = self.t[::step]
        positions = self.curve.position(t_skip)
        T, N, B = self.curve.frenet_frame(t_skip)

        self.plotter.subplot(*loc)
        for i, (pos, binorm) in enumerate(zip(positions, B)):
            end = pos + binorm * scale
            self.plotter.add_mesh(pv.Line(pos, end), color='purple',
                                  line_width=2, opacity=0.7)

    def add_osculating_circles(self, step: int = 30, loc=(0, 0)):
        """Добавить соприкасающиеся окружности"""
        t_skip = self.t[::step]
        positions = self.curve.position(t_skip)
        T, N, B = self.curve.frenet_frame(t_skip)
        radii = self.curve.radius_of_curvature(t_skip)

        self.plotter.subplot(*loc)
        for j, (pos, norm, radius) in enumerate(zip(positions, N, radii)):
            if radius < np.inf and radius > 0.01:
                center = pos + norm * radius
                # Создаем окружность вручную
                circle_t = np.linspace(0, 2 * np.pi, 50)
                # Проектируем на плоскость нормалей
                circle_points = center[:, np.newaxis] + radius * (
                        np.cos(circle_t) * T[j, :, np.newaxis] +
                        np.sin(circle_t) * B[j, :, np.newaxis]
                )
                circle_poly = pv.Spline(circle_points.T)
                self.plotter.add_mesh(circle_poly, color='orange',
                                      opacity=0.5, line_width=1)

    def add_curvature_plot(self, loc=(0, 1)):
        """Добавить график кривизны"""
        kappa = self.curve.curvature(self.t)

        self.plotter.subplot(*loc)
        positions = self.curve.position(self.t)
        poly = pv.Spline(positions)
        poly['curvature'] = kappa

        self.plotter.add_mesh(poly, scalars='curvature',
                              cmap='viridis', line_width=3,
                              scalar_bar_args={'title': 'Curvature κ'})

        self.plotter.add_text(
            f"κ: min={np.min(kappa):.4f}, max={np.max(kappa):.4f}",
            font_size=10, position=(0, 0))

    def add_speed_plot(self, loc=(1, 0)):
        """Добавить визуализацию скорости"""
        speed = self.curve.speed(self.t)

        self.plotter.subplot(*loc)
        positions = self.curve.position(self.t)
        poly = pv.Spline(positions)
        poly['speed'] = speed

        self.plotter.add_mesh(poly, scalars='speed',
                              cmap='cool', line_width=3,
                              scalar_bar_args={'title': 'Speed'})

        self.plotter.add_text(
            f"Speed: min={np.min(speed):.4f}, max={np.max(speed):.4f}",
            font_size=10, position=(0, 0))

    def add_torsion_plot(self, loc=(1, 1)):
        """Добавить график кручения"""
        torsion = self.curve.torsion(self.t)

        self.plotter.subplot(*loc)
        positions = self.curve.position(self.t)
        poly = pv.Spline(positions)
        poly['torsion'] = np.abs(torsion)  # берем абсолютное значение

        self.plotter.add_mesh(poly, scalars='torsion',
                              cmap='RdBu', line_width=3,
                              scalar_bar_args={'title': 'Torsion |τ|'})

        self.plotter.add_text(
            f"Torsion: min={np.min(torsion):.4f}, max={np.max(torsion):.4f}",
            font_size=10, position=(0, 0))

    def show(self):
        """Показать визуализацию"""
        self.plotter.show()