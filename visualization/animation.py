import numpy as np
import pyvista as pv
import time
import threading
from typing import Callable


class AnimationEngine:
    """Чистый движок анимации с бесконечным циклом"""

    def __init__(self, curve=None, num_frames: int = 300, frame_delay: float = 0.05, speed: float = 1.0):
        """
        Args:
            curve: Curve3D объект (опционально)
            num_frames: количество кадров в одном цикле
            frame_delay: задержка между кадрами в секундах
            speed: скорость проигрывания (не используется в новой версии)
        """
        self.curve = curve
        self.num_frames = num_frames
        self.frame_delay = frame_delay
        self.speed = speed
        self.current_t = 0.0
        self.stop_event = threading.Event()
        self.calculation_thread = None
        self.frame_count = 0  # Для отладки

    def start(self):
        """Запустить расчеты"""
        print("🎬 Поток расчетов запущен")

        self.stop_event.clear()
        self.frame_count = 0
        self.calculation_thread = threading.Thread(
            target=self._calculation_loop, daemon=True
        )
        self.calculation_thread.start()

    def _calculation_loop(self):
        """Цикл расчетов - работает бесконечно"""
        frame = 0
        try:
            while not self.stop_event.is_set():
                # Зацикливаем от 0 до 1
                self.current_t = (frame % self.num_frames) / self.num_frames
                self.frame_count = frame

                frame += 1
                time.sleep(self.frame_delay)
        finally:
            print(f"🛑 Поток расчетов остановлен (всего кадров: {self.frame_count})")

    def stop(self):
        """Остановить расчеты"""
        self.stop_event.set()
        if self.calculation_thread and self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=1.0)

    def get_fps(self) -> float:
        """Получить текущий FPS"""
        if self.frame_delay > 0:
            return 1.0 / self.frame_delay
        return 0.0


class CurveVisualizer:
    """Визуализация кривой"""

    def __init__(self, curve, engine, window_size=(1000, 800), hide_radius: float = 0.15):
        """
        Args:
            curve: объект кривой
            engine: AnimationEngine
            window_size: размер окна
            hide_radius: радиус скрытия траектории вокруг стрелки (0-1)
        """
        self.curve = curve
        self.engine = engine
        self.window_size = window_size
        self.hide_radius = hide_radius  # ★ Новый параметр

        self.plotter = None
        self.render_thread = None
        self.stop_event = threading.Event()

        from visualization.actor_manager import ActorManager
        self.actor_manager = ActorManager()
        self.on_update: Callable = self.actor_manager.update_all

        # ★ Для отрисовки траектории
        self._trajectory_actors = None

    def add_actor(self, actor):
        """Добавить актор"""
        self.actor_manager.add_actor(actor)

    def remove_actor(self, actor):
        """Удалить актор"""
        self.actor_manager.remove_actor(actor)

    def _get_trajectory_mesh(self, t_current: float) -> list:
        """
        Получить части траектории без участка под стрелкой

        Returns:
            список (points1, points2) для двух частей траектории
        """
        # Вычисляем интервал для скрытия
        t_start = max(0, t_current - self.hide_radius)
        t_end = min(1, t_current + self.hide_radius)

        # Разбиваем траекторию на части
        # Часть 1: от 0 до t_start
        t_part1 = np.linspace(0, t_start, max(2, int(300 * t_start)))
        positions_part1 = self.curve.position(t_part1)

        # Часть 2: от t_end до 1
        t_part2 = np.linspace(t_end, 1, max(2, int(300 * (1 - t_end))))
        positions_part2 = self.curve.position(t_part2)

        return [positions_part1, positions_part2]

    def _render_loop(self):
        """Цикл рендеринга"""
        print("🎨 Поток рендеринга запущен")

        # Создаем плоттер
        self.plotter = pv.Plotter(window_size=self.window_size)
        self.plotter.set_background("black")

        # ★ Добавляем полную траекторию один раз (для справки)
        # Можно убрать, если не нужна
        # t_values = np.linspace(0, 1, 300)
        # positions = self.curve.position(t_values)
        # self.plotter.add_mesh(
        #     pv.lines_from_points(positions),
        #     color="yellow",
        #     line_width=3,
        #     opacity=0.3
        # )

        self.plotter.show(interactive_update=True, auto_close=False)
        print("🖼️ Плоттер инициализирован\n")

        # Переменные для отслеживания
        last_t = -1
        trajectory_actors = [None, None]  # ★ Два актора для двух частей

        # Цикл рендеринга
        try:
            iren = self.plotter.iren
            while not self.stop_event.is_set():
                try:
                    current_t = self.engine.current_t

                    # ★ Обновляем траекторию только если t изменился значительно
                    if abs(current_t - last_t) > 0.005:
                        # Удаляем старые части траектории
                        for actor in trajectory_actors:
                            if actor is not None:
                                try:
                                    self.plotter.remove_actor(actor)
                                except:
                                    pass

                        # Получаем новые части
                        trajectory_parts = self._get_trajectory_mesh(current_t)

                        # Рисуем первую часть (до стрелки)
                        if len(trajectory_parts[0]) > 1:
                            mesh1 = pv.lines_from_points(trajectory_parts[0])
                            trajectory_actors[0] = self.plotter.add_mesh(
                                mesh1,
                                color="yellow",
                                line_width=3
                            )

                        # Рисуем вторую часть (после стрелки)
                        if len(trajectory_parts[1]) > 1:
                            mesh2 = pv.lines_from_points(trajectory_parts[1])
                            trajectory_actors[1] = self.plotter.add_mesh(
                                mesh2,
                                color="yellow",
                                line_width=3
                            )

                        last_t = current_t

                    # Обновляем акторы (стрелки)
                    if self.on_update:
                        self.on_update(self.plotter, current_t)

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

    def show(self):
        """Запустить визуализацию"""
        self.render_thread = threading.Thread(
            target=self._render_loop, daemon=False
        )
        self.render_thread.start()

        if self.render_thread.is_alive():
            self.render_thread.join()

    def stop(self):
        """Остановить визуализацию"""
        self.stop_event.set()
