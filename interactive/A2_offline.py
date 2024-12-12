import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

from taichi_tracer.renderer import A2Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data

import numpy as np
import matplotlib.pyplot as plt
import taichi.math as tm

from tqdm import tqdm

def main():
    scene_data = load_scene_data(SceneName.SPECULAR_SPHERES, EnvironmentName.STUDIO)
    renderer = A2Renderer(scene_data=scene_data, width=512, height=512)

    #renderer.set_sample_uniform()
    #renderer.set_sample_brdf()
    renderer.set_sample_microfacet()

    spp = 100
    #for _ in range(spp):
    for _ in tqdm(range(spp), desc="Rendering Image"):
        renderer.render()

    img = renderer.canvas.to_numpy()
    img = np.rot90(np.clip(img, 0, 1))

    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    main()

