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

# ★ Правильно: передаем num_frames и frame_delay
engine = AnimationEngine(num_frames=300, frame_delay=0.05)
visualizer = CurveVisualizer(curve, engine)

# Добавляем стрелки
visualizer.add_actor(ArrowActor(curve, "tangent", scale=0.2, color="red"))
visualizer.add_actor(ArrowActor(curve, "normal", scale=0.2, color="blue"))
visualizer.add_actor(ArrowActor(curve, "binormal", scale=0.2, color="green"))

# Запускаем
engine.start()
visualizer.show()
engine.stop()