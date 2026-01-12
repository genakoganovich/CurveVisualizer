import numpy as np
from core.curve import Curve3D
from visualization.animation import ThreadedAnimation


def create_circle():
    """Создать простую окружность"""
    t = np.linspace(0, 2 * np.pi, 200)

    x = np.cos(t)
    y = np.sin(t)
    z = np.zeros_like(t)  # Плоская кривая в плоскости XY

    return np.column_stack([x, y, z])


def main():
    print("=" * 60)
    print("🎬 CurveVisualizer - Простая анимация")
    print("=" * 60)

    # Создаем окружность
    points = create_circle()
    curve = Curve3D(points)

    # Информация о кривой
    print(f"\n✅ Кривая 'Circle' загружена")
    print(f"   Длина кривой: {curve.total_length:.3f}")

    # Запускаем анимацию
    print(f"\n▶️ Запуск анимации...")
    print("   Закройте окно для завершения программы\n")

    animation = ThreadedAnimation(curve, num_frames=200, frame_delay=0.05)
    animation.start()

    print("\n" + "=" * 60)
    print("✅ Программа завершена")
    print("=" * 60)


if __name__ == '__main__':
    main()