import taichi as ti
import taichi.math as tm
import numpy as np

from .ray import Ray


@ti.data_oriented
class Camera:

    def __init__(self, width: int = 128, height: int = 128) -> None:

        # Camera pixel width and height are fixed
        self.width = width
        self.height = height

        # Camera parameters that can be modified are stored as fields
        self.eye = ti.Vector.field(n=3, shape=(), dtype=float)
        self.at = ti.Vector.field(n=3, shape=(), dtype=float)
        self.up = ti.Vector.field(n=3, shape=(), dtype=float)
        self.fov = ti.field(shape=(), dtype=float)

        self.x = ti.Vector.field(n=3, shape=(), dtype=float)
        self.y = ti.Vector.field(n=3, shape=(), dtype=float)
        self.z = ti.Vector.field(n=3, shape=(), dtype=float)

        self.camera_to_world = ti.Matrix.field(n=4, m=4, shape=(), dtype=float)

        # Initialize with some default params
        self.set_camera_parameters(
            eye=tm.vec3([0, 0, 5]),
            at=tm.vec3([0, 0, 0]),
            up=tm.vec3([0, 1, 0]),
            fov=60.
            )
        
        self.aspect_ratio = self.width / self.height
        fov_rad = self.fov[None] * tm.pi / 180
        self.fov_correction = tm.tan(fov_rad / 2.0)


    def set_camera_parameters(
        self, 
        eye: tm.vec3 = None, 
        at: tm.vec3 = None, 
        up: tm.vec3 = None, 
        fov: float = None
        ) -> None:

        if eye: self.eye[None] = eye
        if at: self.at[None] = at
        if up: self.up[None] = up
        if fov: self.fov[None] = fov
        self.compute_matrix()


    @ti.kernel
    def compute_matrix(self):

        eye = self.eye[None]
        at = self.at[None]
        up = self.up[None]

        z = (at - eye).normalized()
        x = (up.cross(z)).normalized()
        y = (z.cross(x)).normalized()

        self.x[None] = x
        self.y[None] = y
        self.z[None] = z

        self.camera_to_world[None] = tm.mat4(
            tm.vec4([x, 0.0]),
            tm.vec4([y, 0.0]),
            tm.vec4([z, 0.0]),
            tm.vec4([eye, 1.0]),
        ).transpose()


    @ti.func
    def generate_ray(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> Ray:
        
        
        # ndc coords are in the range [-1, 1]
        # camera coords have a been scaled by the fov and aspect ratio
        # we want to shift the coord by a random amount to simulate jitter
        # to do this wi use ti.random() to generate a random number in the range [-0.5, 0.5]



        ndc_coords =  self.generate_ndc_coords(pixel_x, pixel_y, jitter)
        camera_coords = self.generate_camera_coords(ndc_coords)
        
        ray = Ray()
        ray.origin = self.eye[None]
        d = self.camera_to_world[None] @ camera_coords
        ray.direction = tm.vec3(d[0],d[1],d[2]).normalized()

        
        return ray


    @ti.func
    def generate_ndc_coords(self, pixel_x: int, pixel_y: int, jitter: bool = False) -> tm.vec2:
        # cast to float to avoid integer division
        # pixel_x_center = ti.cast(pixel_x, float) + 0.5
        # pixel_y_center = ti.cast(pixel_y, float) + 0.5
        # x_jitter = ti.random() - 0.5 # between -0.5 and 0.5
        # y_jitter = ti.random() - 0.5 # between -0.5 and 0.5
        # x = pixel_x_center + x_jitter
        # y = pixel_y_center + y_jitter

        # more cleanly

        x = pixel_x + jitter * ti.random()
        y = pixel_y + jitter * ti.random()

        ndc_x, ndc_y = (x / self.width) * 2 - 1, (y / self.height) * 2 -1


        return tm.vec2([ndc_x, ndc_y])

    @ti.func
    def generate_camera_coords(self, ndc_coords: tm.vec2) -> tm.vec4:
        cam_x = ndc_coords[0] * self.aspect_ratio * self.fov_correction
        cam_y = ndc_coords[1] * self.fov_correction
        cam_z = 1.0

        return tm.vec4([cam_x, cam_y, cam_z, 0.0])