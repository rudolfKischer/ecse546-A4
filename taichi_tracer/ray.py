import taichi as ti
import taichi.math as tm


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3


@ti.dataclass
class HitData:
    is_hit: bool
    is_backfacing: bool
    triangle_id: int
    distance: float
    barycentric_coords: tm.vec2
    normal: tm.vec3
    material_id : int