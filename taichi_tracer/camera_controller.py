import time

import taichi as ti
import taichi.math as tm

from .camera import Camera


class CameraController:

    def __init__(
        self,
        camera: Camera,
        window: ti.ui.Window,
        sensitivity: float = 1.5
        ):

        self.camera = camera
        self.window = window
        self.sensitivity = sensitivity
        self.last_time = None


    def update(self) -> bool:
        # Use elapsed time to calibrate movement speed
        if self.last_time is None:
            self.last_time = time.perf_counter_ns()
        time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
        speed = self.sensitivity * time_elapsed
        self.last_time = time.perf_counter_ns()

        # Grab the camera basis
        x = self.camera.x[None]
        y = self.camera.y[None]
        z = self.camera.z[None]

        # Update the camera position (eye)
        eye_delta = tm.vec3(0)

        # Zoom in/out
        if self.window.is_pressed("e"):
            eye_delta += speed * z
        if self.window.is_pressed("q"):
            eye_delta -= speed * z

        # Pan up/down/left/right
        if self.window.is_pressed("w"):
            eye_delta += speed * y
        if self.window.is_pressed("s"):
            eye_delta -= speed * y
        if self.window.is_pressed("a"):
            eye_delta -= speed * x
        if self.window.is_pressed("d"):
            eye_delta += speed * x

        # Update the camera view direction (at)
        at_delta = tm.vec3(0)

        # Pan up/down/left/right
        if self.window.is_pressed(ti.ui.UP):
            at_delta += speed * y
        if self.window.is_pressed(ti.ui.DOWN):
            at_delta -= speed * y
        if self.window.is_pressed(ti.ui.LEFT):
            at_delta -= speed * x
        if self.window.is_pressed(ti.ui.RIGHT):
            at_delta += speed * x

        # Set the params and trigger re-compute of camera basis and inverse look at matrix
        self.camera.set_camera_parameters(
            eye = self.camera.eye[None] + eye_delta,
            at = self.camera.at[None] + at_delta,
            )
        
        # Check if an update happened
        update_happened = not (eye_delta.norm() == 0 and at_delta.norm() == 0)

        return update_happened
