import numpy as np
from core.curve import Curve3D
from visualization.visualizer import CurveVisualizer


def create_helix(turns: int = 5, height: float = 10):
    """Создать спираль"""
    t = np.linspace(0, 1, 200)
    theta = t * turns * 2 * np.pi

    x = np.cos(theta)
    y = np.sin(theta)
    z = t * height

    return np.column_stack([x, y, z])


def main():
    points = create_helix(turns=5, height=10)
    curve = Curve3D(points)

    viz = CurveVisualizer(curve, num_samples=500)

    # Геометрия (полная)
    viz.add_trajectory(loc=viz.ax_geom, color='blue', linewidth=3)
    viz.add_tangent_vectors(step=30, scale=0.5, loc=viz.ax_geom)
    viz.add_normal_vectors(step=30, scale=0.3, loc=viz.ax_geom)
    viz.add_binormal_vectors(step=30, scale=0.3, loc=viz.ax_geom)
    viz.add_osculating_circles(step=20, loc=viz.ax_geom)

    viz.plotter.subplot(*viz.ax_geom)
    viz.plotter.set_position([0, 0, 0])
    viz.plotter.set_focus([0, 0, 5])
    viz.plotter.add_title("Geometry: Frenet Frame (T, N, B)")

    # Кривизна
    viz.add_curvature_plot(loc=viz.ax_curv)
    viz.plotter.subplot(*viz.ax_curv)
    viz.plotter.add_title("Curvature κ(t)")

    # Кинематика
    viz.add_speed_plot(loc=viz.ax_kin)
    viz.plotter.subplot(*viz.ax_kin)
    viz.plotter.add_title("Speed |dP/dt|")

    # Кручение
    viz.add_torsion_plot(loc=viz.ax_tor)
    viz.plotter.subplot(*viz.ax_tor)
    viz.plotter.add_title("Torsion τ(t)")

    viz.show()


if __name__ == '__main__':
    main()