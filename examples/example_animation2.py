from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.actors import (
    ArrowActor,
    RadiusOfCurvatureActor,
    CurvatureActor,
    TorsionActor,
    SpeedActor
)
import numpy as np

t = np.linspace(0, 4*np.pi, 100)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (4*np.pi)
])
curve = Curve3D(points)

engine = AnimationEngine(num_frames=300, frame_delay=0.08)
visualizer = CurveVisualizer(curve, engine)

# ★ Теперь smoothing работает для всех акторов!
visualizer.add_actor(ArrowActor(curve, "tangent", scale=0.2, color="red", smoothing=0.7))
visualizer.add_actor(ArrowActor(curve, "normal", scale=0.2, color="green", smoothing=0.7))
visualizer.add_actor(ArrowActor(curve, "binormal", scale=0.2, color="blue", smoothing=0.7))

visualizer.add_actor(RadiusOfCurvatureActor(curve, scale=0.5, color="cyan", opacity=0.3, smoothing=0.7))
visualizer.add_actor(CurvatureActor(curve, scale=0.3, color="magenta", smoothing=0.7))
visualizer.add_actor(TorsionActor(curve, scale=0.3, color="orange", smoothing=0.7))
visualizer.add_actor(SpeedActor(curve, scale=0.3, color="lime", smoothing=0.7))

engine.start()
visualizer.show()
engine.stop()