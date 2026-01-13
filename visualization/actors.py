# visualization/actors.py
import numpy as np
import pyvista as pv
from visualization.base_actor import BaseActor


class ArrowActor(BaseActor):
    """Стрелка на кривой (касательная, нормаль, бинормаль)"""

    def __init__(self, curve, arrow_type: str = "tangent", scale: float = 0.3,
                 color: str = "white", smoothing: float = 0.0):
        super().__init__(curve, color, smoothing)
        self.arrow_type = arrow_type
        self.scale = scale
        self._direction_func = self._get_direction_func()

    def _get_direction_func(self):
        methods = {
            "tangent": self.curve.tangent,
            "normal": self.curve.normal,
            "binormal": self.curve.binormal,
        }
        method = methods.get(self.arrow_type)
        if method is None:
            raise ValueError(f"Unknown arrow_type: {self.arrow_type}")
        return method

    def _compute_geometry(self, t: float) -> tuple:
        position = self.curve.position(np.array([t]))[0]
        direction = self._direction_func(np.array([t]))[0]
        direction = direction / (np.linalg.norm(direction) + 1e-10) * self.scale
        return position, direction

    def _create_mesh_geometry(self, position: np.ndarray, direction: np.ndarray):
        """★ Создает меш БЕЗ добавления в plotter"""
        return pv.Arrow(start=position, direction=direction, scale=0.1)

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """★ Создает и добавляет меш в plotter (первый раз)"""
        arrow = self._create_mesh_geometry(position, direction)
        return plotter.add_mesh(arrow, color=self.color)


class CurvatureActor(BaseActor):
    """Стрелка кривизны"""

    arrow_type = "curvature"

    def __init__(self, curve, scale: float = 0.5, color: str = "magenta", smoothing: float = 0.0):
        super().__init__(curve, color, smoothing)
        self.scale = scale

    def _compute_geometry(self, t: float) -> tuple:
        position = self.curve.position(np.array([t]))[0]
        curvature = self.curve.curvature(np.array([t]))[0]
        _, normal, _ = self.curve.frenet_frame(np.array([t]))
        normal = normal[0]
        direction = normal * curvature * self.scale
        return position, direction

    def _create_mesh_geometry(self, position: np.ndarray, direction: np.ndarray):
        """★ Создает меш БЕЗ добавления в plotter"""
        return pv.Arrow(start=position, direction=direction, scale=0.1)

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """★ Создает и добавляет меш в plotter (первый раз)"""
        arrow = self._create_mesh_geometry(position, direction)
        return plotter.add_mesh(arrow, color=self.color)


class TorsionActor(BaseActor):
    """Стрелка кручения"""

    arrow_type = "torsion"

    def __init__(self, curve, scale: float = 0.5, color: str = "orange", smoothing: float = 0.0):
        super().__init__(curve, color, smoothing)
        self.scale = scale

    def _compute_geometry(self, t: float) -> tuple:
        position = self.curve.position(np.array([t]))[0]
        torsion = self.curve.torsion(np.array([t]))[0]
        _, _, binormal = self.curve.frenet_frame(np.array([t]))
        binormal = binormal[0]
        direction = binormal * abs(torsion) * self.scale
        return position, direction

    def _create_mesh_geometry(self, position: np.ndarray, direction: np.ndarray):
        """★ Создает меш БЕЗ добавления в plotter"""
        return pv.Arrow(start=position, direction=direction, scale=0.1)

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """★ Создает и добавляет меш в plotter (первый раз)"""
        arrow = self._create_mesh_geometry(position, direction)
        return plotter.add_mesh(arrow, color=self.color)


class SpeedActor(BaseActor):
    """Стрелка скорости"""

    arrow_type = "speed"

    def __init__(self, curve, scale: float = 0.3, color: str = "lime", smoothing: float = 0.0):
        super().__init__(curve, color, smoothing)
        self.scale = scale

    def _compute_geometry(self, t: float) -> tuple:
        position = self.curve.position(np.array([t]))[0]
        velocity = self.curve.velocity(np.array([t]))[0]
        direction = velocity / (np.linalg.norm(velocity) + 1e-10) * self.scale
        return position, direction

    def _create_mesh_geometry(self, position: np.ndarray, direction: np.ndarray):
        """★ Создает меш БЕЗ добавления в plotter"""
        return pv.Arrow(start=position, direction=direction, scale=0.08)

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """★ Создает и добавляет меш в plotter (первый раз)"""
        arrow = self._create_mesh_geometry(position, direction)
        return plotter.add_mesh(arrow, color=self.color)


class RadiusOfCurvatureActor(BaseActor):
    """Окружность кривизны"""

    arrow_type = "radius_of_curvature"

    def __init__(self, curve, scale: float = 1.0, color: str = "cyan",
                 opacity: float = 0.3, smoothing: float = 0.0):
        super().__init__(curve, color, smoothing)

        self.scale = scale
        self.opacity = opacity
        self._last_radius = None
        self._last_normal = None
        self._last_binormal = None

    def _compute_geometry(self, t: float) -> tuple:
        """Вычислить параметры окружности"""
        position = self.curve.position(np.array([t]))[0]
        radius = self.curve.radius_of_curvature(np.array([t]))[0]

        if np.isinf(radius) or radius > 100:
            radius = 10
        radius *= self.scale

        _, normal, binormal = self.curve.frenet_frame(np.array([t]))
        normal = normal[0]
        binormal = binormal[0]

        return (position, (radius, normal, binormal))

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray, plotter):
        """Dummy метод (не используется, переопределяем update)"""
        return None

    def update(self, plotter, t: float):
        """Переопределяем update для окружности"""
        position, (radius, normal, binormal) = self._compute_geometry(t)

        # ★ Сглаживаем радиус
        if self._last_radius is None:
            self._last_radius = radius
        else:
            radius = self._last_radius * self.smoothing + radius * (1 - self.smoothing)
            self._last_radius = radius

        # ★ Сглаживаем нормали
        if self._last_normal is None:
            self._last_normal = normal
            self._last_binormal = binormal
        else:
            normal = self._smooth_value(normal, self._last_normal)
            binormal = self._smooth_value(binormal, self._last_binormal)
            self._last_normal = normal
            self._last_binormal = binormal

        # Центр окружности
        center = position + normal * radius

        # Генерируем точки окружности
        angles = np.linspace(0, 2 * np.pi, 32)
        circle_points = np.zeros((len(angles), 3))

        for i, angle in enumerate(angles):
            circle_points[i] = (
                    center +
                    radius * np.cos(angle) * normal +
                    radius * np.sin(angle) * binormal
            )

        circle_points = np.vstack([circle_points, circle_points[0]])

        # ★ Удаляем старый актор
        if self._actor is not None:
            try:
                plotter.remove_actor(self._actor)
            except:
                pass

        # ★ Создаем новый
        mesh = pv.lines_from_points(circle_points)
        self._actor = plotter.add_mesh(
            mesh,
            color=self.color,
            line_width=2,
            opacity=self.opacity
        )


class EvoluteActor(BaseActor):
    """Эволюта - кривая центров окружностей кривизны"""

    arrow_type = "evolute"

    def __init__(self, curve, color: str = "purple", line_width: int = 2,
                 opacity: float = 0.8, smoothing: float = 0.0):
        """
        Args:
            curve: объект Curve3D
            color: цвет линии
            line_width: толщина линии
            opacity: прозрачность (0-1)
            smoothing: коэффициент сглаживания
        """
        super().__init__(curve, color, smoothing)
        self.line_width = line_width
        self.opacity = opacity
        self._evolute_actor = None

    def _compute_geometry(self, t: float) -> tuple:
        return (None, None)

    def _create_mesh(self, position, direction, plotter):
        return None

    def update(self, plotter, t: float):
        """Рисует эволюту от 0 до текущей точки t"""
        # ★ Генерируем точки эволюты только ДО текущей точки t
        t_values = np.linspace(0, t, max(2, int(150 * t)))  # ← Важно!
        positions = self.curve.position(t_values)
        radii = self.curve.radius_of_curvature(t_values)

        # Получаем Frenet frame для каждой точки
        _, normals, _ = self.curve.frenet_frame(t_values)

        # Центры кривизны
        evolute_points = positions + normals * radii[:, np.newaxis]

        # Удаляем бесконечности
        evolute_points = evolute_points[np.isfinite(evolute_points).all(axis=1)]

        if len(evolute_points) > 1:
            if self._evolute_actor is not None:
                try:
                    plotter.remove_actor(self._evolute_actor)
                except:
                    pass

            mesh = pv.lines_from_points(evolute_points)
            self._evolute_actor = plotter.add_mesh(
                mesh,
                color=self.color,
                line_width=self.line_width,
                opacity=self.opacity
            )