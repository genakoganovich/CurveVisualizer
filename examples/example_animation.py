from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.actors import ArrowActor
import numpy as np

# Кривая (спираль)
t = np.linspace(0, 2*np.pi, 50)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (2*np.pi)
])
curve = Curve3D(points)

# ★ Теперь работает правильно!
# engine = AnimationEngine(curve, speed=0.5, num_frames=300)
engine = AnimationEngine(num_frames=600, frame_delay=0.016)  # ~60 FPS
visualizer = CurveVisualizer(curve, engine)

# Добавляем стрелки
visualizer.add_actor(ArrowActor(curve, "tangent", scale=0.2, color="red"))
visualizer.add_actor(ArrowActor(curve, "normal", scale=0.2, color="green"))
visualizer.add_actor(ArrowActor(curve, "binormal", scale=0.2, color="blue"))

# Запуск
engine.start()
visualizer.show()
engine.stop()