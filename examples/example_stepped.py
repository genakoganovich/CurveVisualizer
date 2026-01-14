from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.animation_modes import AnimationMode
from visualization.actors import ArrowActor
import numpy as np

t = np.linspace(0, 4*np.pi, 100)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (4*np.pi)
])
curve = Curve3D(points)

# ★ РЕЖИМ 2: Касательная движется с шагом
engine = AnimationEngine(num_frames=120, frame_delay=0.5)
visualizer = CurveVisualizer(
    curve,
    engine,
    mode=AnimationMode.STEPPED,
    num_steps=12  # ★ 12 шагов
)

visualizer.add_actor(
    ArrowActor(curve, "tangent", scale=0.3, color="red", smoothing=0.0)
)

print("📍 Режим 2: Касательная движется с шагом (12 шагов)")
engine.start()
visualizer.show()
engine.stop()