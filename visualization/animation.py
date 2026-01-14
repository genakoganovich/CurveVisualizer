# visualization/animation.py
import pyvista as pv
import numpy as np
import threading
import time
from typing import Callable
from visualization.animation_modes import AnimationMode


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
        self.frame_count = 0
        self.start_time = None

    def start(self):
        """Запустить расчеты"""
        print("🎬 Поток расчетов запущен")

        self.stop_event.clear()
        self.frame_count = 0
        self.start_time = time.time()
        self.calculation_thread = threading.Thread(
            target=self._calculation_loop, daemon=True
        )
        self.calculation_thread.start()

    def _calculation_loop(self):
        """Цикл расчетов - работает бесконечно"""
        frame = 0
        try:
            while not self.stop_event.is_set():
                self.current_t = (frame % self.num_frames) / self.num_frames
                self.frame_count = frame
                frame += 1
                time.sleep(self.frame_delay)
        finally:
            elapsed = time.time() - self.start_time
            print(f"🛑 Поток расчетов остановлен (всего кадров: {self.frame_count}, прошло: {elapsed:.1f}с)")

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

    def get_elapsed_time(self) -> float:
        """Получить прошедшее время с начала анимации"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class CurveVisualizer:
    """Визуализация кривой"""

    def __init__(self, curve, engine, window_size=(1000, 800), mode: AnimationMode = AnimationMode.CONTINUOUS,
                 num_steps: int = 10):
        """
        Args:
            curve: объект кривой
            engine: AnimationEngine
            window_size: размер окна
            mode: режим анимации (CONTINUOUS, STEPPED, ACCUMULATED)
            num_steps: количество шагов для STEPPED и ACCUMULATED режимов
        """
        self.curve = curve
        self.engine = engine
        self.window_size = window_size
        self.mode = mode
        self.num_steps = num_steps

        self.plotter = None
        self.render_thread = None
        self.stop_event = threading.Event()

        from visualization.actor_manager import ActorManager
        self.actor_manager = ActorManager()
        self.on_update: Callable = self.actor_manager.update_all

        self._trajectory_actor = None
        self._last_step_index = -1
        self._accumulated_actors = []

        self._last_stepped_t = None
        self._update_count = 0

    def add_actor(self, actor):
        """Добавить актор"""
        self.actor_manager.add_actor(actor)

    def remove_actor(self, actor):
        """Удалить актор"""
        self.actor_manager.remove_actor(actor)

    def _render_loop(self):
        """Цикл рендеринга"""
        print(f"🎨 Поток рендеринга запущен (режим: {self.mode.value}, шаги: {self.num_steps})")

        # Создаем плоттер
        self.plotter = pv.Plotter(window_size=self.window_size)
        self.plotter.set_background("black")

        # ★ Добавляем полную траекторию один раз
        t_values = np.linspace(0, 1, 300)
        positions = self.curve.position(t_values)
        self.plotter.add_mesh(
            pv.lines_from_points(positions),
            color="yellow",
            line_width=3
        )

        self.plotter.show(interactive_update=True, auto_close=False)
        print("🖼️ Плоттер инициализирован\n")

        # Цикл рендеринга
        try:
            iren = self.plotter.iren
            while not self.stop_event.is_set():
                try:
                    current_t = self.engine.current_t

                    # ★ Обработка в зависимости от режима
                    if self.mode == AnimationMode.CONTINUOUS:
                        self._update_continuous(current_t)
                    elif self.mode == AnimationMode.STEPPED:
                        self._update_stepped(current_t)
                    elif self.mode == AnimationMode.ACCUMULATED:
                        self._update_accumulated(current_t)

                    iren.process_events()
                    self.plotter.render()
                    time.sleep(0.016)

                except RuntimeError:
                    break

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                self.plotter.close()
            except:
                pass

        print("🛑 Поток рендеринга остановлен")

    def _update_continuous(self, current_t: float):
        """★ Режим 1: Касательная движется плавно"""
        if self.on_update:
            self.on_update(self.plotter, current_t)

    def _update_stepped(self, current_t: float):
        """★ Режим 2: Касательная движется с шагом"""
        step_size = 1.0 / self.num_steps

        # ★ Определяем текущий шаг
        stepped_t = round(current_t / step_size) * step_size
        stepped_t = min(stepped_t, 1.0)

        # ★ ОПТИМИЗАЦИЯ: обновляем только если шаг изменился
        if stepped_t != self._last_stepped_t:
            self._last_stepped_t = stepped_t
            self._update_count += 1

            # ★ Логирование с временем
            elapsed = self.engine.get_elapsed_time()
            step_number = int(stepped_t / step_size) + 1

            print(
                f"⏱️  [{elapsed:6.2f}s] STEPPED: t={current_t:.3f} → stepped_t={stepped_t:.3f} (шаг {step_number}/{self.num_steps}) [обновление #{self._update_count}]")

            if self.on_update:
                self.on_update(self.plotter, stepped_t)

    def _update_accumulated(self, current_t: float):
        """★ Режим 3: Добавляются новые касательные с шагом"""
        step_size = 1.0 / self.num_steps

        # ★ Определяем на каком шаге мы сейчас
        current_step_index = int(current_t / step_size)
        if current_t >= 1.0:
            current_step_index = self.num_steps - 1

        # ★ Если перешли на новый шаг
        if current_step_index > self._last_step_index:
            self._last_step_index = current_step_index
            step_t = current_step_index * step_size
            self._update_count += 1

            # ★ Логирование с временем
            elapsed = self.engine.get_elapsed_time()
            print(
                f"⏱️  [{elapsed:6.2f}s] ACCUMULATED: добавляем касательную #{current_step_index + 1}/{self.num_steps} на t={step_t:.3f} [обновление #{self._update_count}]")

            # Добавляем новую касательную в этот момент
            self._add_accumulated_tangent(step_t)

        # ★ НЕ обновляем касательные каждый кадр!
        # Они статичны и уже добавлены в plotter

    def _add_accumulated_tangent(self, t: float):
        """★ Добавляет новую касательную на позицию t"""
        from visualization.actors import ArrowActor

        # Создаем новую касательную в этой позиции
        new_tangent = ArrowActor(
            self.curve,
            "tangent",
            scale=0.3,
            color="red",
            smoothing=0.0
        )

        # Сразу обновляем её до нужной позиции
        new_tangent.update(self.plotter, t)

        # Сохраняем в список
        self._accumulated_actors.append(new_tangent)
        print(f"✅ Добавлена касательная #{len(self._accumulated_actors)}")

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