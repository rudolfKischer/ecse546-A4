import taichi as ti
ti.init(arch=ti.gpu, fast_math=False, default_fp=ti.f32, default_ip=ti.i32)
import taichi.math as tm

from taichi_tracer.renderer import A4Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data

ti.init(arch=ti.metal)

def main():
    # 446 Scene
    scene_data = load_scene_data(SceneName.CORNELL_BOX, EnvironmentName.BLACK)
    # 546 Scene
    # scene_data = load_scene_data(SceneName.CORNELL_BOX_CAUSTIC_SPHERE, EnvironmentName.BLACK)
    
    renderer = A4Renderer(scene_data=scene_data, width=512, height=512)
    window = ti.ui.Window("Interactive Renderer", res=renderer.canvas.shape)
    controller = CameraController(renderer.camera, window)

    renderer.camera.set_camera_parameters(
        eye=tm.vec3([0, 1, 3]),
        at=tm.vec3([0, 1, 0]),
        up=tm.vec3([0, 1, 0]),
        fov=60.
        )

    def control_panel(sample_mode):
        window.GUI.begin("Control Panel", x=0., y=0, width=0.5, height=0.5)
        window.GUI.text("***Camera Controls***")
        window.GUI.text("'Eye' param uses 'wasdqe' keys")
        window.GUI.text("'At' param uses arrow keys")
        window.GUI.text(" ")

        window.GUI.text("***Sampling Mode Controls***")
        if window.GUI.button("Implicit" + (" [Active]" if sample_mode == A4Renderer.ShadingMode.IMPLICIT else "")):
            renderer.set_shading_implicit()
        if window.GUI.button("Explicit" + (" [Active]" if sample_mode == A4Renderer.ShadingMode.EXPLICIT else "")):
            renderer.set_shading_explicit()

        renderer.max_bounces[None] = window.GUI.slider_int("max bounces", renderer.max_bounces[None], 0, 20)
        renderer.rr_termination_probabilty[None] = window.GUI.slider_float("RR termination p", renderer.rr_termination_probabilty[None], 0.0, 0.999)

        window.GUI.end()

        if window.get_event() or controller.update():
            renderer.reset()

    while window.running:
        control_panel(sample_mode=renderer.shading_mode[None])
        renderer.render()
        renderer.postprocess()
        window.get_canvas().set_image(renderer.canvas_postprocessed)
        window.show()

if __name__ == "__main__":
    main()
