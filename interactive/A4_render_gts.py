import taichi as ti
ti.init(arch=ti.gpu, fast_math=False, default_fp=ti.f32, default_ip=ti.i32)
import taichi.math as tm

from taichi_tracer.renderer import A4Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    scene_data = load_scene_data(SceneName.CORNELL_BOX, EnvironmentName.BLACK)
    renderer = A4Renderer(scene_data=scene_data, width=1024, height=1024)
    renderer.camera.set_camera_parameters(
        eye=tm.vec3([0, 1, 3]),
        at=tm.vec3([0, 1, 0]),
        up=tm.vec3([0, 1, 0]),
        fov=60.
        )

    prefix_outputpath = ""
    # prefix_outputpath = "Assignments/handouts_f24/A4_handout_figures/"

    renderer.set_shading_implicit()

    spps = [1, 10, 100]
    bounces = [1, 2, 3, 4]

    for spp in spps:
        for bounce in bounces:
            renderer.reset()
            renderer.max_bounces[None] = bounce
            for _ in tqdm(range(spp), desc="Rendering Implicit Path Tracing Image @ {N} bounces x {s} SPP".format(N=bounce, s=spp)):
                renderer.render()
            renderer.postprocess()

            img = renderer.canvas_postprocessed.to_numpy()
            img = np.rot90(np.clip(img, 0, 1))

            title = "{N}_bounce_Implicit_path_tracing_{s}spp.png".format(N=bounce, s=spp)
            title = prefix_outputpath + title
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(title, bbox_inches='tight', pad_inches=0)
            plt.clf()
    
    renderer.set_shading_explicit()
    

    for spp in spps:
        for bounce in bounces:
            renderer.reset()
            renderer.max_bounces[None] = bounce

            for _ in tqdm(range(spp), desc="Rendering Explicit Path Tracing Image @ {N} bounces x {s} SPP".format(N=bounce, s=spp)):
                renderer.render()
            renderer.postprocess()

            img = renderer.canvas_postprocessed.to_numpy()
            img = np.rot90(np.clip(img, 0, 1))

            title = "{N}_bounce_Explicit_path_tracing_{s}spp.png".format(N=bounce, s=spp)
            title = prefix_outputpath + title
            
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(title, bbox_inches='tight', pad_inches=0)
            plt.clf()


    renderer.set_shading_explicit()

    spp = 100
    rr_probabilities = [0.2, 0.4, 0.6, 0.8]
    for rr_prob in rr_probabilities:
        for bounce in bounces:
            renderer.reset()
            renderer.max_bounces[None] = bounce
            renderer.rr_termination_probabilty[None] = rr_prob

            for _ in tqdm(range(spp), desc="Rendering Explicit Path Tracing Image @ {N} bounces x {s} SPP with Russian Roulette probabilty {rr}".format(N=bounce, s=spp, rr=rr_prob)):
                renderer.render()
            renderer.postprocess()

            img = renderer.canvas_postprocessed.to_numpy()
            img = np.rot90(np.clip(img, 0, 1))

            title = "{N}_bounce_Explicit_path_tracing_{s}spp_rr_prob_{rr}.png".format(N=bounce, s=spp, rr=int(rr_prob*100))
            title = prefix_outputpath + title
            
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(title, bbox_inches='tight', pad_inches=0)
            plt.clf()


    scene_data = load_scene_data(SceneName.CORNELL_BOX_CAUSTIC_SPHERE, EnvironmentName.BLACK)
    renderer = A4Renderer(scene_data=scene_data, width=1024, height=1024)
    renderer.camera.set_camera_parameters(
        eye=tm.vec3([0, 1, 3]),
        at=tm.vec3([0, 1, 0]),
        up=tm.vec3([0, 1, 0]),
        fov=60.
        )

    renderer.set_shading_implicit()

    for spp in spps:
        for bounce in bounces:
            renderer.reset()
            renderer.max_bounces[None] = bounce
            for _ in tqdm(range(spp), desc="Rendering Implicit Path Tracing with Caustics Image @ {N} bounces x {s} SPP".format(N=bounce, s=spp)):
                renderer.render()
            renderer.postprocess()

            img = renderer.canvas_postprocessed.to_numpy()
            img = np.rot90(np.clip(img, 0, 1))

            title = "{N}_bounce_caustic_Implicit_path_tracing_{s}spp.png".format(N=bounce, s=spp)
            title = prefix_outputpath + title
            
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(title, bbox_inches='tight', pad_inches=0)
            plt.clf()
    


if __name__ == "__main__":
    main()
