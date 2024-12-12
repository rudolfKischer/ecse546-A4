import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32, fast_math=False)

from taichi_tracer.renderer import A3Renderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data


import numpy as np
import matplotlib.pyplot as plt
import taichi.math as tm

from tqdm import tqdm

def postprocessing(render):
    render = np.power(render, 1.0/2.2)
    return render

def main():
    scene_data = load_scene_data(SceneName.VEACH, EnvironmentName.BLACK)
    renderer = A3Renderer(scene_data=scene_data, width=1024, height=512)
    
    # Initialize with some default params
    renderer.camera.set_camera_parameters(
    eye=tm.vec3([0, 5.5, 4.5]),
    at=tm.vec3([0, 2, -1.5]),
    up=tm.vec3([0, 1, 0]),
    fov=60.
    )

    spps = [1, 10, 100]
    
    
    renderer.set_sample_brdf()

    for spp in spps:
        renderer.reset()
        for _ in tqdm(range(spp), desc='Rendering BRDF Images'):
            renderer.render()
        
        img = renderer.canvas.to_numpy()
        img = postprocessing(img)
        img = np.rot90(np.clip(img, 0, 1))  
        
        title = "brdf_veach_{x}spp.png".format(x=spp)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(title)
        plt.clf()
    
    renderer.set_sample_light()

    for spp in spps:
        renderer.reset()
        for _ in tqdm(range(spp), desc='Rendering Light Images'):
            renderer.render()
        
        img = renderer.canvas.to_numpy()
        img = postprocessing(img)
        img = np.rot90(np.clip(img, 0, 1))  
        
        
        title = "light_veach_{x}spp.png".format(x=spp)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(title)
        plt.clf()
    

    renderer.set_sample_mis()
    mis_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    for spp in spps:
        for w_brdf in mis_weights:
            renderer.mis_pbrdf[None] = w_brdf
            renderer.mis_plight[None] = 1.0 - renderer.mis_pbrdf[None] 
            renderer.reset()
            for _ in tqdm(range(spp), desc='Rendering MIS Images'):
                renderer.render()
            
            img = renderer.canvas.to_numpy()
            img = postprocessing(img)
            img = np.rot90(np.clip(img, 0, 1))  
            
            
            title = "mis_veach_{x}_brdf_{y}_light_{z}spp.png".format(x=int(100*w_brdf), y=int(100*(1.0-w_brdf)), z=spp)
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(title)
            plt.clf()
        

if __name__ == "__main__":
    main()
