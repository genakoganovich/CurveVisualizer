import numpy as np
import pyvista as pv
from core.curve import Curve3D


def visualize_curve_with_osculating_circles(curve, num_frames: int = 16, scale: float = 0.3):
    """
    Визуализирует кривую с соприкасающимися окружностями и эволютой
    (как на красивой картинке)

    Args:
        curve: объект Curve3D
        num_frames: количество соприкасающихся окружностей
        scale: масштаб Frenet frame
    """

    # Создаем плоттер
    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.set_background("white")

    # ★ Рисуем саму кривую (СИНЯЯ)
    t_values = np.linspace(0, 1, 300)
    positions = curve.position(t_values)
    plotter.add_mesh(
        pv.lines_from_points(positions),
        color="blue",
        line_width=2.5,
        label="Кривая"
    )

    # ★ Рисуем эволюту (КРАСНАЯ)
    evolute_points = []
    for t in t_values:
        position = curve.position(np.array([t]))[0]
        radius = curve.radius_of_curvature(np.array([t]))[0]
        _, normal, _ = curve.frenet_frame(np.array([t]))
        normal = normal[0]

        if np.isinf(radius) or radius > 100:
            continue

        evolute_point = position + normal * radius
        evolute_points.append(evolute_point)

    if evolute_points:
        evolute_points = np.array(evolute_points)
        plotter.add_mesh(
            pv.lines_from_points(evolute_points),
            color="red",
            line_width=2.5,
            label="Эволюта"
        )

    # ★ Добавляем соприкасающиеся окружности (ЗЕЛЁНЫЕ) и радиусы кривизны (ГОЛУБЫЕ)
    step_size = 1.0 / num_frames

    print("\n📊 Osculating Circles Visualization")
    print("=" * 90)
    print(f"{'#':<4} {'t':<8} {'Position':<35} {'Radius':<12} {'Curvature':<12}")
    print("-" * 90)

    for i in range(num_frames):
        t = i * step_size

        # ★ Точка на кривой
        position = curve.position(np.array([t]))[0]

        # ★ Frenet frame
        tangent, normal, binormal = curve.frenet_frame(np.array([t]))
        tangent = tangent[0]
        normal = normal[0]
        binormal = binormal[0]

        # ★ Радиус кривизны и кривизна
        radius = curve.radius_of_curvature(np.array([t]))[0]
        curvature = curve.curvature(np.array([t]))[0]

        # Обрабатываем бесконечные радиусы
        if np.isinf(radius) or radius > 100:
            print(f"{i + 1:<4} {t:<8.3f} ({position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f})  "
                  f"{'∞':<12} {curvature:<12.4f}")
            continue

        # ★ Центр соприкасающейся окружности (на эволюте)
        center = position + normal * radius

        # ★ Рисуем соприкасающуюся окружность (в плоскости нормали и бинормали)
        angles = np.linspace(0, 2 * np.pi, 64)
        circle_points = np.array([
            center + radius * np.cos(a) * normal + radius * np.sin(a) * binormal
            for a in angles
        ])

        plotter.add_mesh(
            pv.lines_from_points(circle_points),
            color="green",
            line_width=1,
            opacity=0.7
        )

        # ★ Рисуем радиус кривизны (от точки кривой к центру)
        radius_line = np.array([position, center])
        plotter.add_mesh(
            pv.lines_from_points(radius_line),
            color="cyan",
            line_width=1.5,
            opacity=0.8
        )

        # ★ Рисуем Frenet frame (маленькие стрелки)
        tangent_scaled = tangent / (np.linalg.norm(tangent) + 1e-10) * scale
        normal_scaled = normal / (np.linalg.norm(normal) + 1e-10) * scale
        binormal_scaled = binormal / (np.linalg.norm(binormal) + 1e-10) * scale

        arrow_t = pv.Arrow(start=position, direction=tangent_scaled, scale=0.08)
        plotter.add_mesh(arrow_t, color="red", opacity=0.6)

        arrow_n = pv.Arrow(start=position, direction=normal_scaled, scale=0.08)
        plotter.add_mesh(arrow_n, color="darkgreen", opacity=0.6)

        arrow_b = pv.Arrow(start=position, direction=binormal_scaled, scale=0.08)
        plotter.add_mesh(arrow_b, color="darkblue", opacity=0.6)

        # Логирование
        print(f"{i + 1:<4} {t:<8.3f} ({position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f})  "
              f"{radius:<12.4f} {curvature:<12.4f}")

    print("-" * 90)
    print(f"✅ Добавлено {num_frames} соприкасающихся окружностей\n")

    # ★ Легенда
    plotter.add_legend(loc='upper left', size=(0.25, 0.25))
    plotter.camera.position = (3, 3, 3)
    plotter.show()


# ★ Основной код
if __name__ == "__main__":
    # Спираль
    t = np.linspace(0, 4 * np.pi, 100)
    points = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / (4 * np.pi)
    ])
    curve = Curve3D(points)

    print("\n" + "=" * 90)
    print("🎨 Osculating Circles and Evolute Visualization")
    print("=" * 90)
    print("СИНЯЯ линия      → Исходная кривая")
    print("КРАСНАЯ линия    → Эволюта (центры кривизны)")
    print("ЗЕЛЁНЫЕ окружности → Соприкасающиеся окружности")
    print("ГОЛУБЫЕ отрезки  → Радиусы кривизны")
    print("Малые стрелки    → Frenet frame (T, N, B)")
    print("=" * 90)

    visualize_curve_with_osculating_circles(curve, num_frames=16, scale=0.25)