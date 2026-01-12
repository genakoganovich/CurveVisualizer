import numpy as np
import pyvista as pv
import threading
import time


class SimpleAnimation:
    """Простая неблокирующая анимация"""

    def __init__(self, curve, num_frames=300, frame_delay=0.05):
        self.curve = curve
        self.num_frames = num_frames
        self.frame_delay = frame_delay
        self.current_t = 0.0
        self.stop_event = threading.Event()
        self.calculation_thread = None
        self.render_thread = None
        self.plotter = None

    def _calculation_loop(self):
        """Цикл расчетов"""
        print("🎬 Поток расчетов запущен")

        while not self.stop_event.is_set():
            # Обновляем t
            self.current_t += 1.0 / (self.num_frames - 1)
            if self.current_t > 1.0:
                self.current_t = 0.0

            time.sleep(self.frame_delay)

        print("🛑 Поток расчетов остановлен")

    def _render_loop(self):
        """Цикл рендеринга в отдельном потоке"""
        print("🎨 Поток рендеринга запущен")

        # Создаем плоттер
        self.plotter = pv.Plotter(window_size=(1000, 800))
        self.plotter.set_background("black")

        # Добавляем траекторию
        t_values = np.linspace(0, 1, 300)
        positions = self.curve.position(t_values)
        self.plotter.add_mesh(
            pv.lines_from_points(positions),
            color="yellow",
            line_width=3,
            label="Trajectory"
        )
        self.plotter.add_legend()

        print("🖼️ Плоттер инициализирован\n")

        # Инициализируем окно
        self.plotter.show(interactive_update=True, auto_close=False)

        # ★ ГЛАВНЫЙ ЦИКЛ РЕНДЕРИНГА ★
        try:
            iren = self.plotter.iren
            while not self.stop_event.is_set():
                try:
                    # Получаем текущую позицию
                    t_arr = np.array([self.current_t])
                    pos = self.curve.position(t_arr)[0]
                    tangent = self.curve.tangent(t_arr)[0]

                    # Очищаем старые объекты (кроме траектории)
                    actors_list = list(self.plotter.actors.values())
                    for actor in actors_list[1:]:
                        try:
                            self.plotter.remove_actor(actor, reset_camera=False)
                        except:
                            pass

                    # Добавляем сферу
                    sphere = pv.Sphere(radius=0.12, center=pos)
                    self.plotter.add_mesh(sphere, color="red")

                    # Добавляем стрелку
                    scale = 0.8
                    end_pos = pos + tangent * scale
                    arrow = pv.Line(pos, end_pos)
                    self.plotter.add_mesh(arrow, color="red", line_width=4)

                    # Обновляем окно
                    iren.process_events()
                    self.plotter.render()

                    time.sleep(0.016)  # ~60 FPS

                except RuntimeError:
                    # Окно закрыто
                    break
                except Exception as e:
                    print(f"⚠️ Ошибка: {e}")
                    break

        except Exception as e:
            print(f"❌ Ошибка в рендеринге: {e}")
        finally:
            try:
                self.plotter.close()
            except:
                pass

        print("🛑 Поток рендеринга остановлен")

    def start(self):
        """Запустить анимацию"""
        print("▶️ Запуск анимации...")

        # Запускаем поток расчетов
        self.calculation_thread = threading.Thread(
            target=self._calculation_loop,
            daemon=False
        )
        self.calculation_thread.start()

        # Запускаем поток рендеринга
        self.render_thread = threading.Thread(
            target=self._render_loop,
            daemon=False
        )
        self.render_thread.start()

        print("📊 Запуск цикла обновления\n")

        # Ждем завершения потока рендеринга (пока не закроют окно)
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join()

        # Останавливаем остальные потоки
        self.stop_event.set()

        if self.calculation_thread and self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=2)

        print("\n✅ Окно закрыто")

    def stop(self):
        """Остановить анимацию"""
        self.stop_event.set()


# Использование:
if __name__ == "__main__":
    from core.curve import Curve3D

    # Создаем спираль
    t = np.linspace(0, 1, 200)
    theta = t * 5 * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    z = t * 10
    points = np.column_stack([x, y, z])

    curve = Curve3D(points)

    print("=" * 60)
    print("🎬 Простая анимация PyVista")
    print("=" * 60)
    print(f"✅ Кривая загружена")
    print(f"   Длина: {curve.total_length:.3f}")
    print(f"\n▶️ Запуск анимации...")
    print("   Закройте окно для завершения\n")

    anim = SimpleAnimation(curve, num_frames=300, frame_delay=0.05)
    anim.start()

    print("\n" + "=" * 60)
    print("✅ Программа завершена")
    print("=" * 60)