from abc import ABC, abstractmethod

import taichi as ti
import taichi.math as tm

from .geometry import Geometry
from .materials import Material
from .ray import Ray, HitData


@ti.data_oriented
class RayIntersector(ABC):

    def __init__(self, geometry: Geometry):
        self.EPSILON = 1e-7
        self.geometry = geometry


    @abstractmethod
    @ti.func
    def query_ray(ray: Ray) -> HitData:
        pass


    @ti.func
    def intersect_triangle(self, ray: Ray, triangle_id: int) -> HitData:

        hit_data = HitData()

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]

        # Normals at each vertex
        normal_indices = self.geometry.triangle_normal_ids[triangle_id-1]-1

        normal_0 = self.geometry.normals[normal_indices[0]]
        normal_1 = self.geometry.normals[normal_indices[1]]
        normal_2 = self.geometry.normals[normal_indices[2]]

        # Material of the triangle
        material_id = self.geometry.triangle_material_ids[triangle_id-1]

        d = ray.direction.normalized()
        o = ray.origin

        e_1 = v1 - v0
        e_2 = v2 - v0

        det = e_1.dot(d.cross(e_2))

        hit_data.is_hit = False
        if det != 0:
          det_inv = 1 / det
          o_v0 = (o - v0)
          u = det_inv * (o_v0.dot(d.cross(e_2)))
          v = det_inv * (d.dot(o_v0.cross(e_1)))
          w = 1 - u - v

          e_m = 0.00
          # if our point is not within margin, we dont `
          if (0 + e_m <= u and u <= 1 - e_m) and (0 + e_m <= v and v <= 1 - e_m) and (0 + e_m <= w and w <= 1 - e_m):
              t = det_inv * (e_2.dot(o_v0.cross(e_1)))
              if t > self.EPSILON:
                hit_data.is_hit = True
                hit_data.triangle_id = triangle_id
                hit_data.distance = t
                hit_data.is_backfacing = (det < 0)
                hit_data.barycentric_coords = tm.vec2([u,v])
                hit_data.normal = (w * normal_0 + u * normal_1 + v * normal_2).normalized()
                if hit_data.is_backfacing:
                    hit_data.normal = -hit_data.normal
                hit_data.material_id = material_id


        return hit_data


@ti.data_oriented
class BruteForceRayIntersector(RayIntersector):

    def __init__(self, geometry: Geometry) -> None:
        super().__init__(geometry)


    @ti.func
    def query_ray(self, ray: Ray) -> HitData:

        closest_hit = HitData()
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            hit_data = self.intersect_triangle(ray, triangle_id)

            if hit_data.is_hit:
                if (hit_data.distance < closest_hit.distance) or (not closest_hit.is_hit):
                    closest_hit = hit_data

        return closest_hit