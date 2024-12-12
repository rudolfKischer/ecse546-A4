from typing import Tuple, List

import taichi as ti
import taichi.math as tm
import numpy as np


@ti.data_oriented
class Geometry:
    def __init__(self,
    vertices: np.array,
    triangle_vertex_ids: np.array,
    normals: np.array,
    triangle_normal_ids: np.array,
    triangle_material_ids: np.array,
    texture_coords: np.array = None,
    triangle_texture_coord_ids: np.array = None,
    ) -> None:

        self.n_vertices = vertices.shape[0]
        self.n_triangles = triangle_vertex_ids.shape[0]
        self.n_normals = normals.shape[0]

        self.vertices = ti.Vector.field(3, shape=(self.n_vertices), dtype=float)
        self.vertices.from_numpy(vertices)

        self.triangle_vertex_ids = ti.Vector.field(3, shape=(self.n_triangles), dtype=int)
        self.triangle_vertex_ids.from_numpy(triangle_vertex_ids)

        self.normals = ti.Vector.field(3, shape=(self.n_normals), dtype=float)
        self.normals.from_numpy(normals)

        self.triangle_normal_ids = ti.Vector.field(3, shape=(self.n_triangles), dtype=int)
        self.triangle_normal_ids.from_numpy(triangle_normal_ids)

        self.triangle_material_ids = ti.field(shape=(self.n_triangles), dtype=int)
        self.triangle_material_ids.from_numpy(triangle_material_ids)

        if texture_coords is not None:
            self.has_texture_coords = True
            self.n_texture_coords = texture_coords.shape[0]

            self.texture_coords = ti.Vector.field(2, shape=(self.n_texture_coords), dtype=float)
            self.texture_coords.from_numpy(texture_coords)

            self.triangle_texture_coord_ids = ti.Vector.field(3, shape=(self.n_triangles), dtype=int)
            self.texture_coords.from_numpy(triangle_texture_coord_ids)
        else:
            self.has_texture_coords = False