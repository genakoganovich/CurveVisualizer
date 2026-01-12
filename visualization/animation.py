import numpy as np
import pyvista as pv
import threading
import time
from dataclasses import dataclass
from typing import Union, Dict, Any, Callable, List


@dataclass
class MeshActor:
    """Визуальный элемент на сцене"""
    mesh: pv.Actor
    color: str


@dataclass
class ActorConfig:
    """Конфиг для одного визуального элемента"""
    name: str
    color: str
    mesh_type: str
    mesh_params: Dict[str, Any]


@dataclass
class ActorState:
    """Состояние актора"""
    position: Union[list, tuple, np.ndarray]
    yaw: float


@dataclass
class ActorVisuals:
    """Визуалы актора + его провайдер состояния"""
    name: str
    visuals: List[str]
    state_provider: Callable[[], ActorState]


class CurveVisualizer:
    """Визуализация кривой"""

    def __init__(self, curve, global_config: Dict[str, Any]):
        self.curve = curve
        self.global_config = global_config
        self.plotter = pv.Plotter()
        self._setup_scene()

        self.visuals: Dict[str, MeshActor] = {}
        self.actors: Dict[str, ActorVisuals] = {}

    def _setup_scene(self):
        """Инициализация сцены"""
        t_values = np.linspace(0, 1, 300)
        positions = self.curve.position(t_values)
        self.plotter.set_background("black")
        self.plotter.add_mesh(
            pv.lines_from_points(positions),
            color="yellow",
            line_width=3
        )

    def add_actor_with_provider(self, actor_name: str,
                                visual_configs: List[ActorConfig],
                                state_provider: Callable[[], ActorState]):
        """Добавить актора с провайдером состояния"""
        visual_names = []

        for config in visual_configs:
            if config.mesh_type == "sphere":
                mesh = pv.Sphere(**config.mesh_params)
            elif config.mesh_type == "line":
                mesh = pv.Line(**config.mesh_params)
            elif config.mesh_type == "cone":
                mesh = pv.Cone(**config.mesh_params)
            else:
                mesh = pv.Sphere(radius=0.1)

            visual = self.plotter.add_mesh(mesh, color=config.color)
            self.visuals[config.name] = MeshActor(visual, config.color)
            visual_names.append(config.name)

        self.actors[actor_name] = ActorVisuals(
            name=actor_name,
            visuals=visual_names,
            state_provider=state_provider
        )

    def update_all_actors(self):
        """Обновить состояние всех акторов"""
        for actor in self.actors.values():
            state = actor.state_provider()

            for visual_name in actor.visuals:
                visual = self.visuals[visual_name].mesh
                visual.SetPosition(list(state.position))
                visual.SetOrientation(0, 0, state.yaw)

    def show(self):
        """Показать окно PyVista"""
        self.plotter.show(interactive_update=True)

    def update(self):
        """Обновить один кадр"""
        self.plotter.update()


class ThreadedAnimation:
    """Анимация с разделением: расчеты в потоке, визуализация в отдельном потоке"""

    def __init__(self, curve, num_frames: int = 300,
                 window_size: tuple = (1000, 800),
                 frame_delay: float = 0.05):
        self.curve = curve
        self.num_frames = num_frames
        self.window_size = window_size
        self.frame_delay = frame_delay

        # Состояние анимации
        self.current_t = {"value": 0.0}
        self.stop_event = threading.Event()

        # Потоки
        self.calculation_thread = None
        self.render_thread = None

        # Визуализатор
        self.visualizer = None

    def _create_visualizer(self):
        """Создать визуализатор и добавить актора с провайдером"""
        global_config = {
            "sphere_radius": self.curve.speed(np.array([0.5]))[0] * 0.1,
            "arrow_scale": 0.8,
        }

        self.visualizer = CurveVisualizer(self.curve, global_config)

        # ★ Определяем провайдер состояния ★
        def state_provider():
            t = self.current_t["value"]
            pos = self.curve.position(np.array([t]))[0]
            tangent = self.curve.tangent(np.array([t]))[0]

            # Вычисляем угол для ориентации
            yaw = np.arctan2(tangent[1], tangent[0]) * 180 / np.pi

            return ActorState(position=pos, yaw=yaw)

        # Конфиги визуальных элементов (только стрелка, без шара)
        visual_configs = [
            ActorConfig(
                name="arrow",
                color="red",
                mesh_type="line",
                mesh_params={"pointa": [0, 0, 0], "pointb": [0.5, 0, 0]}
            ),
        ]

        # Добавляем актора
        self.visualizer.add_actor_with_provider(
            actor_name="curve_point",
            visual_configs=visual_configs,
            state_provider=state_provider
        )

    def _calculation_loop(self):
        """★ ЦИКЛ РАСЧЕТОВ (ОТДЕЛЬНЫЙ ПОТОК) ★"""
        print("🎬 Поток расчетов запущен")

        frame = 0
        while not self.stop_event.is_set():
            try:
                # Обновляем t
                self.current_t["value"] = frame / (self.num_frames - 1)
                frame = (frame + 1) % self.num_frames
                time.sleep(self.frame_delay)
            except Exception as e:
                print(f"❌ Ошибка в расчетах: {e}")
                break

        print("🛑 Поток расчетов остановлен")

    def _render_loop(self):
        """★ ЦИКЛ РЕНДЕРИНГА (ОТДЕЛЬНЫЙ ПОТОК) ★"""
        print("🎨 Поток рендеринга запущен")

        # Инициализируем окно
        self.visualizer.plotter.show(
            interactive_update=True,
            auto_close=False,
            window_size=self.window_size
        )

        print("🖼️ Плоттер инициализирован\n")

        # ★ ГЛАВНЫЙ ЦИКЛ РЕНДЕРИНГА ★
        try:
            iren = self.visualizer.plotter.iren
            while not self.stop_event.is_set():
                try:
                    # Обновляем все акторы
                    self.visualizer.update_all_actors()

                    # Обновляем окно
                    iren.process_events()
                    self.visualizer.plotter.render()

                    time.sleep(0.016)  # ~60 FPS

                except RuntimeError:
                    # Окно закрыто
                    break
                except Exception as e:
                    print(f"⚠️ Ошибка рендеринга: {e}")
                    break

        except Exception as e:
            print(f"❌ Ошибка в потоке рендеринга: {e}")
        finally:
            try:
                self.visualizer.plotter.close()
            except:
                pass

        print("🛑 Поток рендеринга остановлен")

    def start(self):
        """Запустить анимацию"""
        print("▶️ Запуск анимации...")

        # Создаем визуализатор
        self._create_visualizer()

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

        # ★ Ждем завершения потока рендеринга (пока не закроют окно) ★
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join()

        # Останавливаем остальные потоки
        self.stop_event.set()

        if self.calculation_thread and self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=2)

        print("\n✅ Анимация завершена")

    def stop(self):
        """Остановить анимацию"""
        self.stop_event.set()
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=2)
        if self.calculation_thread and self.calculation_thread.is_alive():
            self.calculation_thread.join(timeout=2)