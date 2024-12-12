import taichi as ti

from .geometry import Geometry
from .materials import MaterialLibrary
from .environment import Environment
from .ray_intersector import RayIntersector
from .sampler import MeshLightSampler
from .ray import Ray, HitData


class SceneData:

    def __init__(
        self,
        geometry: Geometry,
        material_library: MaterialLibrary,
        environment: Environment,
        ray_intersector: RayIntersector,
        ) -> None:

        self.geometry = geometry
        self.material_library = material_library
        self.environment = environment
        self.ray_intersector = ray_intersector

        # Setup for mesh light sampling
        self.mesh_light_sampler = MeshLightSampler(self.geometry, self.material_library)

        # # Setup for environment light sampling
        # self.environment_light_sampler = EnvironmentLightSampler(self.environment)

    @ti.func
    def query_ray(self, ray: Ray) -> HitData:
        return self.ray_intersector.query_ray(ray)
