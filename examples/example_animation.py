import numpy as np
from core.curve import Curve3D
from visualization.animation import ThreadedAnimation


def create_helix(turns: int = 5, height: float = 10):
    """Создать спираль"""
    t = np.linspace(0, 1, 200)
    theta = t * turns * 2 * np.pi

    x = np.cos(theta)
    y = np.sin(theta)
    z = t * height

    return np.column_stack([x, y, z])


def create_lissajous():
    """Создать фигуру Лиссажу"""
    t = np.linspace(0, 2 * np.pi, 300)

    x = np.sin(3 * t)
    y = np.sin(5 * t)
    z = np.sin(7 * t)

    return np.column_stack([x, y, z])


def create_butterfly():
    """Создать кривую бабочки"""
    t = np.linspace(0, 12 * np.pi, 500)

    x = np.sin(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t))
    y = np.cos(t) * (np.exp(np.cos(t)) - 2 * np.cos(4 * t))
    z = t / (12 * np.pi) * 5  # высота

    return np.column_stack([x, y, z])


def main():
    print("=" * 60)
    print("🎬 CurveVisualizer - Многопоточная анимация")
    print("=" * 60)

    # Выбираем кривую
    print("\nВыберите кривую:")
    print("1. Спираль (Helix)")
    print("2. Фигура Лиссажу")
    print("3. Кривая бабочки")

    choice = input("Введите выбор (1, 2 или 3): ").strip()

    if choice == "1":
        points = create_helix(turns=5, height=10)
        curve_name = "Helix"
    elif choice == "3":
        points = create_butterfly()
        curve_name = "Butterfly"
    else:
        points = create_lissajous()
        curve_name = "Lissajous"

    curve = Curve3D(points)

    # Информация о кривой
    print(f"\n✅ Кривая '{curve_name}' загружена")
    print(f"   Длина кривой: {curve.total_length:.3f}")

    # Расчитываем кривизну
    t_test = np.linspace(0, 1, 100)
    curvatures = curve.curvature(t_test)
    speeds = curve.speed(t_test)

    print(f"   Макс кривизна: {np.max(curvatures):.4f}")
    print(f"   Макс скорость: {np.max(speeds):.4f}")

    # Создаем и запускаем анимацию
    print(f"\n▶️ Запуск анимации для {curve_name}...")
    print("   Используются разные потоки для расчета и визуализации")
    print("   Закройте окно PyVista для завершения программы\n")

    # ★ Создаем анимацию
    animation = ThreadedAnimation(
        curve=curve,
        num_frames=300,
        window_size=(1000, 800),
        frame_delay=0.05  # ~20 FPS для расчетов
    )

    # ★ Запускаем анимацию (блокирует до закрытия окна PyVista)
    animation.start()

    print("\n" + "=" * 60)
    print("✅ Программа завершена")
    print("=" * 60)


if __name__ == '__main__':
    main()