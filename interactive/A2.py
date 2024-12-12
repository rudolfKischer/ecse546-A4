import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

from taichi_tracer.renderer import A2Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data

ti.init(arch=ti.metal)
# ti.init(debug=True)

def main():
    scene_data = load_scene_data(SceneName.SPECULAR_SPHERES, EnvironmentName.STUDIO)
    renderer = A2Renderer(scene_data=scene_data, width=512, height=512)
    window = ti.ui.Window("Interactive Renderer", res=renderer.canvas.shape)
    controller = CameraController(renderer.camera, window)

    def control_panel(sample_mode):
        window.GUI.begin("Control Panel", x=0., y=0, width=0.5, height=0.5)
        window.GUI.text("***Camera Controls***")
        window.GUI.text("'Eye' param uses 'wasdqe' keys")
        window.GUI.text("'At' param uses arrow keys")
        window.GUI.text(" ")

        window.GUI.text("***Sampling Mode Controls***")
        if window.GUI.button("Uniform" + (" [Active]" if sample_mode == A2Renderer.SampleMode.UNIFORM else "")):
            renderer.set_sample_uniform()
        if window.GUI.button("BRDF" + (" [Active]" if sample_mode == A2Renderer.SampleMode.BRDF else "")):
            renderer.set_sample_brdf()
        if window.GUI.button("Microfacet[546 Only]" + (" [Active]" if sample_mode == A2Renderer.SampleMode.MICROFACET else "")):
            renderer.set_sample_microfacet()
        window.GUI.end()

        if window.get_event() or controller.update():
            renderer.reset()


    while window.running:
        control_panel(renderer.sample_mode[None])
        renderer.render()
        window.get_canvas().set_image(renderer.canvas)
        window.show()

if __name__ == "__main__":
    main()

