import numpy as np
import pyvista as pv
import threading
import time


class ThreadedAnimation:
    """Простая анимация кривой со стрелкой"""

    def __init__(self, curve, num_frames: int = 300,
                 window_size: tuple = (1000, 800),
                 frame_delay: float = 0.05):
        self.curve = curve
        self.num_frames = num_frames
        self.window_size = window_size
        self.frame_delay = frame_delay

        self.current_t = 0.0
        self.stop_event = threading.Event()
        self.calculation_thread = None
        self.render_thread = None
        self.plotter = None

    def _calculation_loop(self):
        """Обновляет t"""
        print("🎬 Поток расчетов запущен")

        frame = 0
        while not self.stop_event.is_set():
            self.current_t = frame / (self.num_frames - 1)
            frame = (frame + 1) % self.num_frames
            time.sleep(self.frame_delay)

        print("🛑 Поток расчетов остановлен")

    def _render_loop(self):
        """Рендеринг окна"""
        print("🎨 Поток рендеринга запущен")

        # Создаем плоттер
        self.plotter = pv.Plotter(window_size=self.window_size)
        self.plotter.set_background("black")

        # Добавляем кривую
        t_values = np.linspace(0, 1, 300)
        positions = self.curve.position(t_values)
        self.plotter.add_mesh(
            pv.lines_from_points(positions),
            color="yellow",
            line_width=3
        )

        # Инициализируем окно
        self.plotter.show(interactive_update=True, auto_close=False)
        print("🖼️ Плоттер инициализирован\n")

        # Цикл рендеринга
        try:
            iren = self.plotter.iren
            while not self.stop_event.is_set():
                try:
                    # Получаем текущую позицию и направление
                    t_arr = np.array([self.current_t])
                    pos = self.curve.position(t_arr)[0]
                    tangent = self.curve.tangent(t_arr)[0]

                    # Удаляем старую стрелку (но не кривую)
                    actors_list = list(self.plotter.actors.values())
                    for actor in actors_list[1:]:
                        try:
                            self.plotter.remove_actor(actor, reset_camera=False)
                        except:
                            pass

                    # Добавляем новую стрелку
                    scale = 0.3
                    end_pos = pos + tangent * scale
                    arrow = pv.Line(pos, end_pos)
                    self.plotter.add_mesh(arrow, color="red", line_width=4)

                    # Обновляем окно
                    iren.process_events()
                    self.plotter.render()
                    time.sleep(0.016)

                except RuntimeError:
                    break

        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            try:
                self.plotter.close()
            except:
                pass

        print("🛑 Поток рендеринга остановлен")

    def start(self):
        """Запустить анимацию"""
        print("▶️ Запуск анимации...")

        # Поток расчетов
        self.calculation_thread = threading.Thread(
            target=self._calculation_loop, daemon=False
        )
        self.calculation_thread.start()

        # Поток рендеринга
        self.render_thread = threading.Thread(
            target=self._render_loop, daemon=False
        )
        self.render_thread.start()

        print("📊 Запуск цикла обновления\n")

        # Ждем закрытия окна
        if self.render_thread.is_alive():
            self.render_thread.join()

        # Останавливаем все
        self.stop_event.set()
        if self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=2)

        print("\n✅ Анимация завершена")

    def stop(self):
        """Остановить"""
        self.stop_event.set()