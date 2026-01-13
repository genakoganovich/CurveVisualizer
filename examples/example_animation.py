from core.curve import Curve3D
from visualization.animation import AnimationEngine, CurveVisualizer
from visualization.actors import (
    ArrowActor,
    RadiusOfCurvatureActor,
    EvoluteActor,
    # OsculatingPlaneActor,
    # VelocityFieldActor,
    # FrenetFrameActor,
    # CurvatureCombActor
)
import numpy as np

# Спираль
t = np.linspace(0, 4*np.pi, 100)
points = np.column_stack([
    np.cos(t),
    np.sin(t),
    t / (4*np.pi)
])
curve = Curve3D(points)

engine = AnimationEngine(num_frames=300, frame_delay=0.08)
visualizer = CurveVisualizer(curve, engine)

# Основное (рекомендуется)
visualizer.add_actor(ArrowActor(curve, "tangent", scale=0.2, color="red", smoothing=0.7))
visualizer.add_actor(ArrowActor(curve, "normal", scale=0.2, color="green", smoothing=0.7))
visualizer.add_actor(ArrowActor(curve, "binormal", scale=0.2, color="blue", smoothing=0.7))
visualizer.add_actor(RadiusOfCurvatureActor(curve, scale=0.5, color="cyan", opacity=0.3, smoothing=0.7))

# Дополнения
visualizer.add_actor(EvoluteActor(curve, color="cyan", line_width=2, opacity=1))
# visualizer.add_actor(OsculatingPlaneActor(curve, size=0.3, color="yellow", opacity=0.1, smoothing=0.7))
# visualizer.add_actor(FrenetFrameActor(curve, scale=0.15, smoothing=0.7))  # ⭐⭐⭐

engine.start()
visualizer.show()
engine.stop()