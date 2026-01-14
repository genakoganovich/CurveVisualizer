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

# ★ РЕЖИМ 3: Добавляются новые касательные с шагом
engine = AnimationEngine(num_frames=300, frame_delay=0.05)
visualizer = CurveVisualizer(
    curve,
    engine,
    mode=AnimationMode.ACCUMULATED,
    num_steps=10  # ★ 10 касательных
)

visualizer.add_actor(
    ArrowActor(curve, "tangent", scale=0.3, color="red", smoothing=0.0)
)

print("📍 Режим 3: Добавляются касательные с шагом (10 шагов)")
engine.start()
visualizer.show()
engine.stop()