from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.actors import ArrowActor
import numpy as np

# Создаем кривую
t = np.linspace(0, 2*np.pi, 50)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (2*np.pi)
])
curve = Curve3D(points)

# Движок и визуализатор
engine = AnimationEngine(curve, num_frames=300, frame_delay=0.05)

# ★ Параметр hide_radius управляет размером скрытого участка
visualizer = CurveVisualizer(
    curve,
    engine,
    hide_radius=0.02  # Скрыть ±10% траектории вокруг стрелки
)

# Добавляем стрелки
visualizer.add_actor(ArrowActor(curve, "tangent", scale=0.2, color="red"))
visualizer.add_actor(ArrowActor(curve, "normal", scale=0.2, color="green"))
visualizer.add_actor(ArrowActor(curve, "binormal", scale=0.2, color="blue"))

# Запускаем
engine.start()
visualizer.show()
engine.stop()