import numpy as np
import pyvista as pv
from core.curve import Curve3D


def visualize_curve_with_frenet_frame(curve, num_frames: int = 12, scale: float = 0.3):
    """
    Визуализирует кривую с полным Frenet frame (касательная, нормаль, бинормаль)
    и центрами кривизны (эволютой)

    Args:
        curve: объект Curve3D
        num_frames: количество Frenet frames
        scale: масштаб стрелок
    """

    # Создаем плоттер
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background("black")

    # ★ Рисуем кривую
    t_values = np.linspace(0, 1, 300)
    positions = curve.position(t_values)
    plotter.add_mesh(
        pv.lines_from_points(positions),
        color="yellow",
        line_width=1,
        label="Кривая"
    )

    # ★ Рисуем эволюту
    evolute_points = []
    for t in t_values:
        position = curve.position(np.array([t]))[0]
        radius = curve.radius_of_curvature(np.array([t]))[0]
        _, normal, _ = curve.frenet_frame(np.array([t]))
        normal = normal[0]

        # Обрабатываем бесконечные радиусы
        if np.isinf(radius) or radius > 100:
            continue

        # Центр кривизны = текущая позиция + радиус * нормаль
        evolute_point = position + normal * radius
        evolute_points.append(evolute_point)

    if evolute_points:
        evolute_points = np.array(evolute_points)
        plotter.add_mesh(
            pv.lines_from_points(evolute_points),
            color="purple",
            line_width=2,
            opacity=0.7,
            label="Эволюта"
        )

    # ★ Добавляем Frenet frames с шагом
    step_size = 1.0 / num_frames

    print("\n📊 Frenet Frame with Evolute Visualization")
    print("=" * 80)
    print(f"{'#':<4} {'t':<8} {'Pt':<30} {'Pe':<30} {'Radius':<10}")
    print("-" * 80)

    for i in range(num_frames):
        t = i * step_size

        # ★ Вычисляем позицию на кривой (Pt)
        position = curve.position(np.array([t]))[0]

        # ★ Вычисляем Frenet frame
        tangent, normal, binormal = curve.frenet_frame(np.array([t]))
        tangent = tangent[0]
        normal = normal[0]
        binormal = binormal[0]

        # ★ Вычисляем радиус кривизны и центр кривизны (Pe - точка эволюты)
        radius = curve.radius_of_curvature(np.array([t]))[0]

        # Обрабатываем бесконечные радиусы
        if np.isinf(radius) or radius > 100:
            radius_display = np.inf
            evolute_point = position  # На месте если радиус бесконечен
        else:
            radius_display = radius
            evolute_point = position + normal * radius

        # ★ Нормализуем векторы
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10) * scale
        normal = normal / (np.linalg.norm(normal) + 1e-10) * scale
        binormal = binormal / (np.linalg.norm(binormal) + 1e-10) * scale

        # ★ Красная стрелка - касательная
        arrow_t = pv.Arrow(start=position, direction=tangent, scale=0.1)
        plotter.add_mesh(arrow_t, color="red", opacity=0.9)

        # ★ Зелёная стрелка - нормаль
        arrow_n = pv.Arrow(start=position, direction=normal, scale=0.1)
        plotter.add_mesh(arrow_n, color="green", opacity=0.9)

        # ★ Синяя стрелка - бинормаль
        arrow_b = pv.Arrow(start=position, direction=binormal, scale=0.1)
        plotter.add_mesh(arrow_b, color="blue", opacity=0.9)

        # ★ Соединяем Pt и Pe отрезком (отрезок нормали/радиуса)
        line_points = np.array([position, evolute_point])
        plotter.add_mesh(pv.lines_from_points(line_points),
                         color="cyan", line_width=2, opacity=0.7)

        # Логирование
        print(f"{i + 1:<4} {t:<8.3f} ({position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f})  "
              f"({evolute_point[0]:6.2f}, {evolute_point[1]:6.2f}, {evolute_point[2]:6.2f})  "
              f"{radius_display:<10.3f}")

    print("-" * 80)
    print(f"✅ Добавлено {num_frames} Frenet frames с центрами кривизны\n")

    # ★ Добавляем легенду
    plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[1, 0, 0], scale=0.1),
                     color="red", label="Tangent (T)")
    plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[0, 1, 0], scale=0.1),
                     color="green", label="Normal (N)")
    plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[0, 0, 1], scale=0.1),
                     color="blue", label="Binormal (B)")

    plotter.add_legend(loc='upper right')
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

    print("\n" + "=" * 80)
    print("🎨 Frenet Frame with Evolute Visualization")
    print("=" * 80)
    print("Красные стрелки      → Касательная (Tangent)")
    print("Зелёные стрелки      → Нормаль (Normal)")
    print("Синие стрелки        → Бинормаль (Binormal)")
    print("Голубые отрезки      → Радиусы кривизны (Pt → Pe)")
    print("=" * 80)

    visualize_curve_with_frenet_frame(curve, num_frames=12, scale=0.3)