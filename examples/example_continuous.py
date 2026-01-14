from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.animation_modes import AnimationMode
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

# ★ РЕЖИМ 1: Касательная движется плавно
engine = AnimationEngine(num_frames=300, frame_delay=0.05)
visualizer = CurveVisualizer(curve, engine, mode=AnimationMode.CONTINUOUS)

visualizer.add_actor(
    ArrowActor(curve, "tangent", scale=0.3, color="red", smoothing=0.0)
)

print("📍 Режим 1: Касательная движется плавно")
engine.start()
visualizer.show()
engine.stop()