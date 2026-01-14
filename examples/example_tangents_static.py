import numpy as np
import pyvista as pv
from core.curve import Curve3D


def visualize_curve_with_frenet_frame(curve, num_frames: int = 12, scale: float = 0.3):
    """
    Визуализирует кривую с полным Frenet frame (касательная, нормаль, бинормаль)

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
        line_width=3,
        label="Кривая"
    )

    # ★ Добавляем Frenet frames с шагом
    step_size = 1.0 / num_frames

    print("\n📊 Frenet Frame Visualization")
    print("=" * 60)
    print(f"{'#':<4} {'t':<8} {'Tangent':<20} {'Normal':<20} {'Binormal':<20}")
    print("-" * 60)

    for i in range(num_frames):
        t = i * step_size

        # Вычисляем позицию и Frenet frame
        position = curve.position(np.array([t]))[0]
        tangent, normal, binormal = curve.frenet_frame(np.array([t]))

        tangent = tangent[0]
        normal = normal[0]
        binormal = binormal[0]

        # Нормализуем
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

        # Логирование
        t_norm = np.linalg.norm(tangent)
        n_norm = np.linalg.norm(normal)
        b_norm = np.linalg.norm(binormal)

        print(f"{i + 1:<4} {t:<8.3f} {t_norm:<20.3f} {n_norm:<20.3f} {b_norm:<20.3f}")

    print("-" * 60)
    print(f"✅ Добавлено {num_frames} Frenet frames\n")

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

    print("\n" + "=" * 60)
    print("🎨 Frenet Frame Visualization")
    print("=" * 60)
    print("Красные стрелки   → Касательная (Tangent)")
    print("Зелёные стрелки   → Нормаль (Normal)")
    print("Синие стрелки     → Бинормаль (Binormal)")
    print("=" * 60)

    visualize_curve_with_frenet_frame(curve, num_frames=12, scale=0.3)