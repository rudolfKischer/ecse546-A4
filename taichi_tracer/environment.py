import taichi as ti
import taichi.math as tm
import numpy as np

from .ray_intersector import Ray


@ti.data_oriented
class Environment:
    def __init__(self, image: np.array):

        self.x_resolution = image.shape[0]
        self.y_resolution = image.shape[1]

        # original env map
        self.image = ti.Vector.field(
            n=3, dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # luminance env map             
        # luminance = 0.2126*rgb.x + 0.7152*rgb.y + 0.0722*rgb.z
        self.image_scalar = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # p(theta) marginal
        self.marginal_ptheta = ti.field(
            dtype=float, shape=(self.y_resolution)
        )

        # cdf of p(theta)
        self.cdf_ptheta = ti.field(
            dtype=float, shape=(self.y_resolution)
        )

        # p(phi | theta)
        self.conditional_p_phi_given_theta = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        # cdf of p(phi | theta)
        self.cdf_p_phi_given_theta = ti.field(
            dtype=float, shape=(self.x_resolution, self.y_resolution)
        )

        self.image.from_numpy(image)

        self.intensity = ti.field(dtype=float, shape=())
        self.set_intensity(1.)


    def set_intensity(self, intensity: float) -> None:
        self.intensity[None] = intensity


    @ti.func
    def query_ray(self, ray: Ray) -> tm.vec3:
        ray.direction = ray.direction.normalized()

        atan_denom = tm.sign(ray.direction.x) * max(abs(ray.direction.x), 1e-6)

        u = 0.5 + (0.5 * tm.atan2(ray.direction.z, atan_denom) / tm.pi)
        v = 0.5 + (tm.asin(ray.direction.y) / tm.pi)

        x = int(u * self.x_resolution)
        y = int(v * self.y_resolution)
        return self.image[x, y]

    @ti.kernel
    def precompute_envmap(self):
        self.precompute_scalar()
        self.precompute_marginal_ptheta()
        self.precompute_conditional_p_phi_given_theta()
        self.precompute_cdfs()


    @ti.func
    def precompute_scalar(self):        
        color_correction = tm.vec3([0.2126, 0.7152, 0.0722])
        total = 0.
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):

            v = y / self.y_resolution 
            theta = (v) * tm.pi
            sin_theta = tm.sin(theta)
            self.image_scalar[x, y] = self.image[x,y].dot(color_correction) * sin_theta
            total += self.image_scalar[x, y]

        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):
            self.image_scalar[x, y] /= total


        return

    @ti.func
    def precompute_marginal_ptheta(self):
        total = 0.0
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):
            self.marginal_ptheta[y] += self.image_scalar[x, y]
        
        for y in ti.ndrange(self.y_resolution):
            total += self.marginal_ptheta[y]

        for y in ti.ndrange(self.y_resolution):
            self.marginal_ptheta[y] /= total
        return 

    @ti.func
    def precompute_conditional_p_phi_given_theta(self):
        avg_val = 1.0
        
        for x, y in ti.ndrange(self.x_resolution, self.y_resolution):
            if abs(self.marginal_ptheta[y]) > 0.0000000000000001:  
                self.conditional_p_phi_given_theta[x, y] = self.image_scalar[x, y] / self.marginal_ptheta[y] 
            else:
                self.conditional_p_phi_given_theta[x, y] = 0 
            avg_val += self.conditional_p_phi_given_theta[x, y] / (self.x_resolution * self.y_resolution)

    @ti.func
    def precompute_cdfs(self):
        ptheta_cumulative = 0.
        
        ti.loop_config(serialize=True)
        for y in ti.ndrange(self.y_resolution):
            ptheta_cumulative += self.marginal_ptheta[y] 
            self.cdf_ptheta[y] = ptheta_cumulative

        for y in ti.ndrange(self.y_resolution):
            cumulative = 0.
            for x in ti.ndrange(self.x_resolution):
                cumulative += self.conditional_p_phi_given_theta[x, y]
                self.cdf_p_phi_given_theta[x, y] = cumulative

    @ti.func
    def sample_theta(self, u1: float) -> int:
        pos = 0
        ti.loop_config(serialize=True)
        for y in ti.ndrange(self.y_resolution):
            if self.cdf_ptheta[y] > u1:
                break
            pos = y
        return pos

    @ti.func
    def sample_phi(self, theta: int, u2: float) -> int: 
        pos = 0
        for x in ti.ndrange(self.x_resolution):
            if self.cdf_p_phi_given_theta[x, theta] > u2:
                break
            pos = x
        return pos
    

    @ti.func
    def importance_sample_envmap(self) -> tm.vec2:
        u1 = ti.random()
        u2 = ti.random()

        sampled_theta = self.sample_theta(u1)
        prev_theta = self.cdf_ptheta[sampled_theta - 1] if sampled_theta > 0.0 else 0.0
        theta_cdf = self.cdf_ptheta[sampled_theta]
        theta_diff = theta_cdf - prev_theta
        frac_theta = (u1 - prev_theta) / theta_diff if theta_diff > 1e-6 else 0.0
        lerped_theta = lerp(frac_theta, sampled_theta - 1, sampled_theta)

        sampled_phi = self.sample_phi(sampled_theta, u2)
        prev_phi = self.cdf_p_phi_given_theta[sampled_phi - 1, sampled_theta] if sampled_phi > 0.0 else 0.0
        phi_cdf = self.cdf_p_phi_given_theta[sampled_phi, sampled_theta]
        phi_diff = phi_cdf - prev_phi
        frac_phi = (u2 - prev_phi) / phi_diff if phi_diff > 1e-6 else 0.0
        lerped_phi = lerp(frac_phi, sampled_phi - 1, sampled_phi) 
        u = lerped_phi / float(self.x_resolution)
        v = lerped_theta / float(self.y_resolution)
        
        
        
        # palceholder
        return tm.vec2([u, v])


@ti.func
def lerp(x: float, a: float, b: float) -> float:
    return ((1.0-x) * a) + (x * b)