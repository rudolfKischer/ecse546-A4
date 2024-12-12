import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)
import taichi.math as tm

from taichi_tracer.renderer import A1Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, load_scene_data

def main():
    scene_data = load_scene_data(SceneName.CORNELL_BOX)
    renderer = A1Renderer(scene_data=scene_data, width=512, height=512)
    window = ti.ui.Window("Interactive Renderer", res=renderer.canvas.shape)
    controller = CameraController(renderer.camera, window)

    renderer.camera.set_camera_parameters(
        eye=tm.vec3([0, 1, 3]),
        at=tm.vec3([0, 1, 0]),
        up=tm.vec3([0, 1, 0]),
        fov=60.
        )

    def control_panel(shader_mode):
        window.GUI.begin("Control Panel", x=0., y=0, width=0.5, height=0.5)
        window.GUI.text("***Camera Controls***")
        window.GUI.text("'Eye' param uses 'wasdqe' keys")
        window.GUI.text("'At' param uses arrow keys")
        window.GUI.text(" ")

        window.GUI.text("***Shade Mode Controls***")
        if window.GUI.button("Hit" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.HIT else "")):
            renderer.set_shade_hit()
        if window.GUI.button("Triangle ID" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.TRIANGLE_ID else "")):
            renderer.set_shade_triangle_ID()
        if window.GUI.button("Distance" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.DISTANCE else "")):
            renderer.set_shade_distance()
        if window.GUI.button("Barycentric" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.BARYCENTRIC else "")):
            renderer.set_shade_barycentrics()
        if window.GUI.button("Normal" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.NORMAL else "")):
            renderer.set_shade_normal()
        if window.GUI.button("Material ID" + (" [Active]" if shader_mode == A1Renderer.ShadeMode.MATERIAL_ID else "")):
            renderer.set_shade_material_ID()
        window.GUI.end()


    while window.running:
        controller.update()
        control_panel(renderer.shade_mode[None])
        renderer.render()
        window.get_canvas().set_image(renderer.canvas)
        window.show()

if __name__ == "__main__":
    main()

