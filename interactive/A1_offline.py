import taichi as ti
ti.init(arch=ti.gpu)
import taichi.math as tm

from taichi_tracer.renderer import A1Renderer
from taichi_tracer.scene_data_loader import SceneName, load_scene_data

import numpy as np
import matplotlib.pyplot as plt

def main():
    scene_data = load_scene_data(SceneName.CORNELL_BOX)
    renderer = A1Renderer(scene_data=scene_data, width=512, height=512)
    renderer.set_shade_normal()

    renderer.camera.set_camera_parameters(
        eye=tm.vec3([0, 1, 3]),
        at=tm.vec3([0, 1, 0]),
        up=tm.vec3([0, 1, 0]),
        fov=60.
        )
    renderer.render()
    img = renderer.canvas.to_numpy()
    img = np.rot90(np.clip(img, 0, 1))

    plt.imshow(img)
    plt.show()
    
if __name__ == "__main__":
    main()

