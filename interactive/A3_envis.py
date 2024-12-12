import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

from taichi_tracer.renderer import EnvISRenderer
from taichi_tracer.camera_controller import CameraController
from taichi_tracer.scene_data_loader import SceneName, EnvironmentName, load_scene_data

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import time

def main():
    scene_data = load_scene_data(SceneName.CORNELL_BOX, EnvironmentName.STUDIO)
    renderer = EnvISRenderer(scene_data=scene_data, width=1024, height=512)
    
    renderer.scene_data.environment.precompute_envmap()

    renderer.reset()
    renderer.set_sample_uniform()

    renderer.render_background()
    background = renderer.background.to_numpy()
    background = np.rot90(np.clip(background, 0, 1))

    samples_list = [1_000_000, 10_000_000, 100_000_000]
    
    for samples in samples_list:
        print("========== {x} SAMPLES ==========".format(x=samples))
        title = "env_is_{x}_samples.png".format(x=str(samples))

        renderer.reset()
        renderer.set_sample_uniform()

        t1 = time.time()
        renderer.sample_env(samples=samples)

        uniform_samples = renderer.count_map.to_numpy()
        
        print("========== UNIFORM SAMPLING ==========")
        print("Minimum Count: ", np.min(uniform_samples))
        print("Mean Count: ", np.mean(uniform_samples))
        print("Median Count: ", np.median(uniform_samples))
        print("Maximum Count: ", np.max(uniform_samples))
        print("Runtime: ", time.time()-t1)
        
        uniform_samples /= np.max(uniform_samples)
        uniform_samples = np.rot90(np.clip(uniform_samples, 0, 1))
        uniform_samples = np.repeat(uniform_samples[:, :, None], 3, axis=-1)

        renderer.reset()
        renderer.set_sample_envmap()

        t1 = time.time()
        renderer.sample_env(samples=samples)
        
        importance_samples = renderer.count_map.to_numpy()
        
        print("========== IMPORTANCE SAMPLING ==========")
        print("Minimum Count: ", np.min(importance_samples))
        print("Mean Count: ", np.mean(importance_samples))
        print("Median Count: ", np.median(importance_samples))
        print("Maximum Count: ", np.max(importance_samples))
        print("Runtime: ", time.time()-t1)

        importance_samples /= np.max(importance_samples)
        importance_samples = np.rot90(np.clip(importance_samples, 0, 1))
        importance_samples = np.repeat(importance_samples[:, :, None], 3, axis=-1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(uniform_samples)
        axes[0].set_title('Uniform Samples')
        axes[0].axis('off')

        axes[1].imshow(background)
        axes[1].set_title('Environment Map')
        axes[1].axis('off')

        axes[2].imshow(importance_samples)
        axes[2].set_title('Importance Samples')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(title)
        plt.clf()

if __name__ == "__main__":
    main()
