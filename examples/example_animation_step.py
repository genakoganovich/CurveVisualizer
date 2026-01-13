# examples/example_tangent_steps.py

from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.actors import ArrowActor
import numpy as np

# Спираль
t = np.linspace(0, 4*np.pi, 100)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (4*np.pi)
])
curve = Curve3D(points)

# ★ Используем num_frames как количество дискретных шагов
engine = AnimationEngine(num_frames=60, frame_delay=0.5)  # 12 шагов по 0.5 сек
visualizer = CurveVisualizer(curve, engine)

# ★ Только касательная
visualizer.add_actor(
    ArrowActor(curve, "tangent", scale=0.3, color="red", smoothing=0.0)
)

engine.start()
visualizer.show()
engine.stop()