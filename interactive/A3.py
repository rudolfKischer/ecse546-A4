import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32, fast_math=False)

from taichi_tracer.renderer import A3Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data

# @ti.kernel
# def postprocessing(renderer):
#     renderer_canvas = tm.pow(renderer, tm.vec3(1.0 / 2.2))
#     renderer_canvas = tm.clamp(renderer_canvas, xmin=0.0, xmax=1.0)
#     return renderer_canvas

def main():
    scene_data = load_scene_data(SceneName.VEACH, EnvironmentName.BLACK)
    renderer = A3Renderer(scene_data=scene_data, width=512, height=512)
    window = ti.ui.Window("Interactive Renderer", res=renderer.canvas.shape)
    controller = CameraController(renderer.camera, window)
    
    # Initialize with some default params
    renderer.camera.set_camera_parameters(
    eye=tm.vec3([0, 5.5, 4.5]),
    at=tm.vec3([0, 2, -1.5]),
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
        if window.GUI.button("Uniform" + (" [Active]" if sample_mode == A3Renderer.SampleMode.UNIFORM else "")):
            renderer.set_sample_uniform()
            renderer.reset()

        if window.GUI.button("BRDF" + (" [Active]" if sample_mode == A3Renderer.SampleMode.BRDF else "")):
            renderer.set_sample_brdf()
            renderer.reset()

        if window.GUI.button("Light" + (" [Active]" if sample_mode == A3Renderer.SampleMode.LIGHT else "")):
            renderer.set_sample_light()
            renderer.reset()

        if window.GUI.button("MIS" + (" [Active]" if sample_mode == A3Renderer.SampleMode.MIS else "")):
            renderer.set_sample_mis()
            renderer.reset()

        
        renderer.mis_plight[None] = window.GUI.slider_float("MIS p_light", renderer.mis_plight[None], 0.0, 1.0)
        renderer.mis_pbrdf[None] = 1.0 - renderer.mis_plight[None]

        renderer.mis_pbrdf[None] = window.GUI.slider_float("MIS p_brdf", renderer.mis_pbrdf[None], 0.0, 1.0)
        renderer.mis_plight[None] = 1.0 - renderer.mis_pbrdf[None]


        window.GUI.end()

        if window.get_event() or controller.update():
            renderer.reset()

    while window.running:
        control_panel(sample_mode=renderer.sample_mode[None])
        renderer.render()
        renderer.postprocess()
        window.get_canvas().set_image(renderer.canvas_postprocessed)
        window.show()

if __name__ == "__main__":
    main()
