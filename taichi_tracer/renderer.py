from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF, MicrofacetBRDF
from .materials import Material


@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

# vec contains nan
@ti.func
def has_nan(vec):
    result = False
    for i in ti.static(range(vec.n)):
        if isnan(vec[i]):
            result = True
    return False

@ti.data_oriented
class A1Renderer:

    # Enumerate the different shading modes
    class ShadeMode(IntEnum):
        HIT = 1
        TRIANGLE_ID = 2
        DISTANCE = 3
        BARYCENTRIC = 4
        NORMAL = 5
        MATERIAL_ID = 6

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.scene_data = scene_data

        self.shade_mode = ti.field(shape=(), dtype=int)
        self.set_shade_hit()

        # Distance at which the distance shader saturates
        self.max_distance = 10.

        # Numbers used to generate colors for integer index values
        self.r = 3.14159265
        self.b = 2.71828182
        self.g = 6.62607015


    def set_shade_hit(self):          self.shade_mode[None] = self.ShadeMode.HIT
    def set_shade_triangle_ID(self):  self.shade_mode[None] = self.ShadeMode.TRIANGLE_ID
    def set_shade_distance(self):     self.shade_mode[None] = self.ShadeMode.DISTANCE
    def set_shade_barycentrics(self): self.shade_mode[None] = self.ShadeMode.BARYCENTRIC
    def set_shade_normal(self):       self.shade_mode[None] = self.ShadeMode.NORMAL
    def set_shade_material_ID(self):  self.shade_mode[None] = self.ShadeMode.MATERIAL_ID


    @ti.kernel
    def render(self):
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] = color


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        color = tm.vec3(0)
        if   self.shade_mode[None] == int(self.ShadeMode.HIT):         color = self.shade_hit(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.TRIANGLE_ID): color = self.shade_triangle_id(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.DISTANCE):    color = self.shade_distance(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.BARYCENTRIC): color = self.shade_barycentric(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.NORMAL):      color = self.shade_normal(hit_data)
        elif self.shade_mode[None] == int(self.ShadeMode.MATERIAL_ID): color = self.shade_material_id(hit_data)
        return color
       

    @ti.func
    def shade_hit(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            if not hit_data.is_backfacing:
                color = tm.vec3(1)
            else: 
                color = tm.vec3([0.5,0,0])
        return color


    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*self.r % 1
            g = triangle_id*self.g % 1
            b = triangle_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color


    @ti.func
    def shade_distance(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            d = tm.clamp(hit_data.distance / self.max_distance, 0,1)
            color = tm.vec3(d)
        return color


    @ti.func
    def shade_barycentric(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            u = hit_data.barycentric_coords[0]
            v = hit_data.barycentric_coords[1]
            w = 1. - u - v
            color = tm.vec3(u,v,w)
        return color


    @ti.func
    def shade_normal(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            normal = hit_data.normal
            color = (normal + 1.) / 2.  # Scale to range [0,1]
        return color


    @ti.func
    def shade_material_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        if hit_data.is_hit:
            material_id = hit_data.material_id + 1 # Add 1 so that ID 0 is not black
            r = material_id*self.r % 1
            g = material_id*self.g % 1
            b = material_id*self.b % 1
            color = tm.vec3(r,g,b)
        return color

@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-3

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()

        self.BRDF = BRDF()
        self.MicrofacetBRDF = MicrofacetBRDF()
        self.UniformSampler = UniformSampler()


    def set_sample_uniform(self):    self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_brdf(self):       self.sample_mode[None] = self.SampleMode.BRDF
    def set_sample_microfacet(self): self.sample_mode[None] = self.SampleMode.MICROFACET


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            
            self.canvas[x,y] += (color - self.canvas[x,y]) / self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)

    @ti.func
    def shade_triangle_id(self, hit_data: HitData) -> tm.vec3:
        color = tm.vec3(0)
        mat = self.scene_data.material_library.materials[hit_data.material_id]
        diffuse = mat.Kd
        if hit_data.is_hit:
            triangle_id = hit_data.triangle_id + 1 # Add 1 so that ID 0 is not black
            r = triangle_id*diffuse[0] % 1
            g = triangle_id*diffuse[1] % 1
            b = triangle_id*diffuse[2] % 1
            color = tm.vec3(r,g,b)
        return color
    
    @ti.func
    def stable_pow(self, base: float, exponent: float) -> float:
        # need to be safe from over and underflow
        intermediate = base
        fast_exponent = tm.log2(exponent)
        for _ in range(int(fast_exponent)):
            if intermediate < 1e-6:
                break
            if intermediate > 1e6:
                break
            intermediate *= intermediate
          
        return intermediate

    
    @ti.func
    def phong_specular_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_s = mat.Kd
        alpha = mat.Ns
        w_r = (2 * w_o.dot(x.normal) * x.normal - w_o).normalized()
        wr_dot_wi = max(0.0,w_r.dot(w_i))
        # if wr_dot_wi > (1. - 1e-6):
            # wr_dot_wi = 1. - 1e-6


        # lobe_term = tm.pow(wr_dot_wi, alpha) # -> leads to underflow when alpha is large. wr_dot_wi <= 1
        # lobe_term = 0
        # lobe_term = self.stable_pow(wr_dot_wi, alpha)
        lobe_term = tm.pow(wr_dot_wi, alpha)
        if lobe_term > 1.0:
            lobe_term = 1.0

        # print the
        if alpha > 60 and lobe_term > 1.0:
            print(f'alpha: {alpha}, lobe_term: {lobe_term:.5e} wr_dot_wi: {wr_dot_wi:.5e}')
            
        specular = (rho_s * (alpha + 1.) / (2. * tm.pi)) * max(lobe_term,0.0)
        return specular
    
    @ti.func
    def phong_diffuse_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_d = mat.Kd
        diffuse = rho_d / tm.pi
        return diffuse

    @ti.func
    def phong_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        alpha = mat.Ns
        result = tm.vec3(0)
        if alpha <= 1:
            result = self.phong_diffuse_brdf(x, w_i, w_o)
        else:
            result = self.phong_specular_brdf(x, w_i, w_o)
        return result
        

    @ti.func
    def visibility(self, x: HitData, r_o: Ray) -> tm.vec3:
        hit_data = self.scene_data.ray_intersector.query_ray(r_o)
        env_map = self.scene_data.environment
        result = env_map.query_ray(r_o)
        # result = tm.vec3(0)
        mat = self.scene_data.material_library.materials[hit_data.material_id]
        if hit_data.is_hit: 
        #    result = tm.vec3(0)
            result = mat.Ke
        # if the data is no hit, set distance to infinity
        # if not hit_data.is_hit:
        #     hit_data.distance = float('inf')
        return result
    
    @ti.func
    def reflect(self, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        w_i = w_i.normalized()
        normal = normal.normalized()
        return (2 * w_i.dot(normal) * normal - w_i).normalized()


    @ti.func
    def shade_ray_uniform(self, ray: Ray, hit_data: HitData, mat: Material) -> tm.vec3:
        env_map = self.scene_data.environment
        w_i = -ray.direction
        u_sampler = self.UniformSampler
        w_o = u_sampler.sample_direction()
        # w_o is in the on the sphere, but we want it to be the normal hemisphere
        # we can do this by taking the dot product of the normal and the direction
        # if the dot product is negative, we flip the direction across the normal
        normal = hit_data.normal
        if w_o.dot(normal) < 0:
          w_o = -w_o

        cos_theta = max(w_o.dot(normal), 0)
        p_u = u_sampler.evaluate_probability() * 2 # multiply by 2 because we are sampling hemisphere, not sphere

        #incoming light
        p = hit_data.distance * ray.direction + ray.origin
        r_i = Ray(p + 10 * self.RAY_OFFSET * w_o, w_o.normalized())
        l_i = env_map.query_ray(r_i
                              )
        V_x = self.visibility(hit_data, r_i)
        color = V_x  * self.phong_brdf(hit_data, w_i, w_o) * cos_theta / p_u
        L_e = mat.Ke
        color += L_e
        # if has_nan(color):
        #     # set to red if nan
        #     color = tm.vec3(1,0,0)
        # if color[0] == 0.0 and color[1] == 0.0 and color[2] == 0.0:
        #     color = tm.vec3(1,0,0)
        return color


    @ti.func
    def shade_ray_brdf(self, ray: Ray, hit_data: HitData, mat: Material) -> tm.vec3:
        env_map = self.scene_data.environment
        w_i = -ray.direction.normalized()
        normal = hit_data.normal.normalized()
        brdf = self.BRDF
        w_o = brdf.sample_direction(mat, w_i, normal).normalized()

        p = hit_data.distance * ray.direction.normalized() + ray.origin
        r_i = Ray(p + self.RAY_OFFSET * w_o, w_o.normalized())
        #l_i = env_map.query_ray(r_i)
        l_e = mat.Ke
        # l_i = l_e
        V_x = self.visibility(hit_data, r_i)
        f_r = self.phong_brdf(hit_data, w_i, w_o)
        # color = brdf.evaluate_brdf(mat, w_o, w_i, hit_data.normal.normalized()) * V_x + l_e 
        p_brdf = brdf.evaluate_probability(mat, w_o, w_i, normal)
        cos_theta = max(w_o.dot(normal), 0)

        color = tm.vec3(0.0)
        a = mat.Ns
        if p_brdf != 0.0:
          color = (f_r) * V_x * cos_theta / (p_brdf)

        b = a
        if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
            rho_s = mat.Kd
            color = V_x * cos_theta * rho_s 


        return color

    @ti.func
    def shade_ray_microfacet(self, ray: Ray, hit_data: HitData, mat: Material) -> tm.vec3: 
        env_map = self.scene_data.environment
        w_o = -ray.direction.normalized()
        normal = hit_data.normal.normalized()
        microfacetbrdf = self.MicrofacetBRDF
        w_m = microfacetbrdf.sample_direction(mat, w_o, normal).normalized()
        w_i = self.reflect(w_o, w_m).normalized()



        p = hit_data.distance * ray.direction.normalized() + ray.origin
        r_i = Ray(p + 100 * self.RAY_OFFSET * w_i, w_i.normalized())
        l_i = env_map.query_ray(r_i)
        V_x = self.visibility(hit_data, r_i)



        f_r_microfacet = microfacetbrdf.evaluate_brdf(mat, w_o, w_i, w_m, normal)
        p_microfacet = microfacetbrdf.evaluate_probability(mat, w_o, w_i, w_m, normal)
        cos_theta = max(w_i.dot(normal), 0)

        color = f_r_microfacet / p_microfacet * V_x * cos_theta
        return color


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)


        # if the ray does no hit anything, we return the color of the environment map
        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        mat = self.scene_data.material_library.materials[hit_data.material_id]
        env_map = self.scene_data.environment

        if not hit_data.is_hit:
            color = env_map.query_ray(ray)
        else:
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                color = self.shade_ray_uniform(ray, hit_data, mat)
            elif self.sample_mode[None] == int(self.SampleMode.BRDF):
                color =  self.shade_ray_brdf(ray, hit_data, mat)
            elif self.sample_mode[None] == int(self.SampleMode.MICROFACET):
                color = self.shade_ray_microfacet(ray, hit_data, mat)
        return color


@ti.data_oriented
class EnvISRenderer:
    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        ENVMAP = 2
    
    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.width = width
        self.height = height
        
        self.camera = Camera(width=width, height=height)
        self.count_map = ti.field(dtype=float, shape=(width, height))
        
        self.background = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.scene_data = scene_data
        self.sample_mode = ti.field(shape=(), dtype=int)

        self.set_sample_uniform()




    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_envmap(self):    
        self.sample_mode[None] = self.SampleMode.ENVMAP

    @ti.func
    def render_background(self, x: int, y: int) -> tm.vec3:
        uv_x, uv_y = float(x)/self.width, float(y)/self.height
        uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
        
        background = self.scene_data.environment.image[int(uv_x), int(uv_y)]
            

        return background


    @ti.kernel
    def render_background(self):
        for x,y in ti.ndrange(self.width, self.height):
            uv_x, uv_y = float(x)/float(self.width), float(y)/float(self.height)
            uv_x, uv_y = uv_x*self.scene_data.environment.x_resolution, uv_y*self.scene_data.environment.y_resolution
            color = self.scene_data.environment.image[int(uv_x), int(uv_y)]

            self.background[x,y] = color

    @ti.kernel
    def sample_env(self, samples: int):
        for _ in ti.ndrange(samples):
            if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
                x = int(ti.random() * self.width)
                y = int(ti.random() * self.height)


                self.count_map[x,y] += 1.0
                
            elif self.sample_mode[None] == int(self.SampleMode.ENVMAP):
                sampled_phi_theta = self.scene_data.environment.importance_sample_envmap()
                x = sampled_phi_theta[0] * self.width
                y = sampled_phi_theta[1] * self.height

                self.count_map[int(x), int(y)] += 1.0
    
    @ti.kernel
    def reset(self):
        self.count_map.fill(0.)


@ti.data_oriented
class A3Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        LIGHT = 3
        MIS = 4

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        self.a2_renderer = A2Renderer(width=self.width, height=self.height, scene_data=self.scene_data)

        self.mis_plight = ti.field(dtype=float, shape=())
        self.mis_pbrdf = ti.field(dtype=float, shape=())

        self.mis_plight[None] = 0.5
        self.mis_pbrdf[None] = 0.5

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self): 
        self.sample_mode[None] = self.SampleMode.UNIFORM
        self.a2_renderer.set_sample_uniform()
    def set_sample_brdf(self):    
        self.sample_mode[None] = self.SampleMode.BRDF
        self.a2_renderer.set_sample_brdf()
    def set_sample_light(self):    self.sample_mode[None] = self.SampleMode.LIGHT
    def set_sample_mis(self):    self.sample_mode[None] = self.SampleMode.MIS


    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]
    
    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)
    
    @ti.func
    def shade_ray_light_sampling(self, ray: Ray, hit_data: HitData, mat: Material) -> tm.vec3:

        color = tm.vec3(0)

        mls = self.scene_data.mesh_light_sampler

        hit_point = ray.origin + ray.direction.normalized() * hit_data.distance


        light_direction, light_triangle_id  = mls.sample_direction(hit_point)
        
        p_t = mls.evaluate_probability() # uniform probability of emissive points 

        shadow_ray = Ray(hit_point + self.RAY_OFFSET * light_direction, light_direction)
        light_hit_data = self.scene_data.ray_intersector.query_ray(shadow_ray)

        shadow_triangle_id = light_hit_data.triangle_id

        if shadow_triangle_id == light_triangle_id:

            w_j = shadow_ray.direction.normalized()
            
            light_distance = 1.
            ny_dot_wj = 1.
            n_y = light_hit_data.normal.normalized()
            n_x = hit_data.normal.normalized()
            nx_dot_wj = max(n_x.dot(w_j), 0)

            light_hit_mat = self.scene_data.material_library.materials[light_hit_data.material_id]
            hit_light_ke = light_hit_mat.Ke
            light_hit_luminance = hit_light_ke.norm()

            light_distance = light_hit_data.distance
            ny_dot_wj = max(n_y.dot(-w_j), 0)
            #if light_hit_data.is_hit and light_hit_luminance > 0.01 and light_triangle_id != -1:




            w_o = w_j 
            w_i = -ray.direction.normalized()
            
            
            fr = self.a2_renderer.phong_brdf(hit_data, w_o, w_i)

            V_x = self.a2_renderer.visibility(hit_data, shadow_ray)

            k_e = mat.Ke

            color = (fr * nx_dot_wj * ny_dot_wj * V_x) / (p_t * light_distance**2)
            color += k_e

        return color 


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)

        hit_data = self.scene_data.ray_intersector.query_ray(ray)
        mat = self.scene_data.material_library.materials[hit_data.material_id]

        L_e = mat.Ke


        if self.sample_mode[None] == int(self.SampleMode.UNIFORM) or self.sample_mode[None] == int(self.SampleMode.BRDF):
            # Uniform or BRDF just calls the A2 renderer
            color = self.a2_renderer.shade_ray(ray)
        else:
            if self.sample_mode[None] == int(self.SampleMode.LIGHT):

                color = self.shade_ray_light_sampling(ray, hit_data, mat)

            if self.sample_mode[None] == int(self.SampleMode.MIS):

                X = ti.random()

                # SINGLE SAMPLE MULTIPLE IMPORTANCE SAMPLING

                # for composite distribution functions like f(x) = f_a(x) * f_b(x) 
                # we migh not be able to easily draw from the inverse probability distribution
                # often we can only draw from pdf_a or pdf_b 
                # this means we can draw low a sample with low probability, but still a non low value of f(x)
                # this will yield very large values for our sample, ultimately leading to high variance
                # which means on average it can take longer to converge

                # we can instead sample from both distributions, and reweight our samples to account for this
                # w_a(x) * f_a(x) / p_a(x) + w_b(x) * f_b(x) / p_b(x)

                # we can use the balance heuristic for this which is
                # w_i(x) = (n_i p_i(x)) / (sum_j n_j p_j(x))

                # F_i = w_i(x) * f_i(x) / p_i(x) 
                # = (p_i / (sum p_j(x)) * (f_i(x) / p_i(x))
                # = f_i(x) / (sum_j p_j(x))

                # for a one sample estimate, we can alternate each sampling strategy
                # we could do 50/50 alternation, or we could set a weighted strategy (note this is a different wieght) then w_i()
                # we will call this q_i
                # then we can sample according to these weights to choose our strategy

                # once we have picked our strategy, we can sample from it
                # but we also divide by the strategy weight q_i

                # F_i = f_i(x) /  ((sum_j p_j(x)) * q_i)






                # if X < self.mis_plight[None]:
                #     color = self.shade_ray_light_sampling(ray, hit_data, mat)
                # else:
                #     color = self.a2_renderer.shade_ray_brdf(ray, hit_data, mat)

                brdf = self.a2_renderer.BRDF
                mls = self.a2_renderer.scene_data.mesh_light_sampler
                normal = hit_data.normal
                w_i = -ray.direction.normalized()
                hit_point = ray.origin + ray.direction.normalized() * hit_data.distance

                w_o = tm.vec3(0)
                q_i = self.mis_pbrdf[None]

                light_triangle_id = -1

                if X < self.mis_pbrdf[None]:
                    w_o = brdf.sample_direction(mat, w_i, normal)
                    q_i = self.mis_pbrdf[None]
                    # TODO: need to account for the other factors in the light brdf
                    # 1. distance to the light
                    # 2. light cosine term
                else:
                    w_o, light_triangle_id = mls.sample_direction(hit_point)
                    q_i = self.mis_plight[None]

                shadow_ray = Ray(hit_point + self.RAY_OFFSET * w_o, w_o)


                
                # x: HitData, w_i, w_o,
                f_r = self.a2_renderer.phong_brdf(hit_data, w_i, w_o)

                p_l = mls.evaluate_probability()
                p_brdf = brdf.evaluate_probability(mat, w_o, w_i, normal)

                p_f = p_l * self.mis_plight[None] + p_brdf * self.mis_pbrdf[None]

                V_x = self.a2_renderer.visibility(hit_data, shadow_ray)

                cosine = max(w_i.dot(normal), 0)

                color = (V_x * f_r * cosine) / (p_f)

                if X > self.mis_pbrdf[None]:
                    shadow_hit = self.a2_renderer.scene_data.ray_intersector.query_ray(shadow_ray)
                    color = (f_r * cosine) / (p_f)

                    if shadow_hit.triangle_id == light_triangle_id and shadow_hit.distance > 0.0001:
                    
                        light_normal = shadow_hit.normal
                        distance = shadow_hit.distance



                        inv_square_dist = 1. / (distance**2)
                        w_j = -shadow_ray.direction.normalized()
                        cosine_light = max(light_normal.dot(w_j), 0)
                        
                        # get hit data luminance
                        light_hit_mat = self.a2_renderer.scene_data.material_library.materials[shadow_hit.material_id]
                        hit_light_ke = light_hit_mat.Ke
                        light_hit_luminance = hit_light_ke.norm()


                        # if shadow_hit.is_hit and light_hit_luminance > 0.0001:
                        color *= cosine_light * inv_square_dist

                            # color the light by mesh light color
                        color *= hit_light_ke
                          
                    else:
                        color = tm.vec3(0)

                L_e = mat.Ke

                color += L_e
                






                # if X < self.mis_plight[None]:
                #     color = self.shade_ray_light_sampling(ray, hit_data, mat) / self.mis_plight[None]
                # else:
                #     color = self.a2_renderer.shade_ray_brdf(ray, hit_data, mat) / self.mis_pbrdf[None]

                #color *= self.mis_plight[None]

        if L_e.norm() > 0.0001:
            color = L_e


        return color


@ti.data_oriented
class A4Renderer:

    # Enumerate the different sampling modes
    class ShadingMode(IntEnum):
        IMPLICIT = 1
        EXPLICIT = 2

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-4

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.canvas_postprocessed = ti.Vector.field(n=3, dtype=float, shape=(width, height))

        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data
        
        self.max_bounces = ti.field(dtype=int, shape=())
        self.max_bounces[None] = 5

        self.rr_termination_probabilty = ti.field(dtype=float, shape=())
        self.rr_termination_probabilty[None] = 0.0

        self.shading_mode = ti.field(shape=(), dtype=int)
        self.set_shading_implicit()

        self.BRDF = BRDF()
        self.UNIFORM = UniformSampler()
        self.MLS = self.scene_data.mesh_light_sampler

    def set_shading_implicit(self): self.shading_mode[None] = self.ShadingMode.IMPLICIT
    def set_shading_explicit(self): self.shading_mode[None] = self.ShadingMode.EXPLICIT

    @ti.kernel
    def postprocess(self):
        for x,y in ti.ndrange(self.width, self.height):
            self.canvas_postprocessed[x, y] = tm.pow(self.canvas[x, y], tm.vec3(1.0 / 2.2))
            self.canvas_postprocessed[x, y] = tm.clamp(self.canvas_postprocessed[x, y], xmin=0.0, xmax=1.0)

    @ti.func
    def phong_specular_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_s = mat.Kd
        alpha = mat.Ns
        w_r = (2 * w_o.dot(x.normal) * x.normal - w_o).normalized()
        wr_dot_wi = max(0.0,w_r.dot(w_i))
        lobe_term = tm.pow(wr_dot_wi, alpha)
        if lobe_term > 1.0:
            lobe_term = 1.0
        specular = (rho_s * (alpha + 1.) / (2. * tm.pi)) * max(lobe_term,0.0)
        return specular
    
    @ti.func
    def phong_diffuse_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        rho_d = mat.Kd
        diffuse = rho_d / tm.pi
        return diffuse

    @ti.func
    def phong_brdf(self, x: HitData, w_i: tm.vec3, w_o: tm.vec3) -> tm.vec3:
        mat = self.scene_data.material_library.materials[x.material_id]
        alpha = mat.Ns
        result = tm.vec3(0)
        if alpha <= 1:
            result = self.phong_diffuse_brdf(x, w_i, w_o)
        else:
            result = self.phong_specular_brdf(x, w_i, w_o)
        # result = self.phong_specular_brdf(x, w_i, w_o)
        return result
    
    @ti.func
    def shade_ray_brdf(self, w_i: tm.vec3, w_o: tm.vec3,  hit_data: HitData, mat: Material) -> tm.vec3:
        normal = hit_data.normal.normalized()
        brdf = self.BRDF

        f_r = self.phong_brdf(hit_data, w_i, w_o)
        p_brdf = brdf.evaluate_probability(mat, w_o, w_i, normal)
        cos_theta = max(w_o.dot(normal), 0)
        # cos_theta = 1.0

        color = tm.vec3(0.0)
        if p_brdf != 0.0:
          color = (f_r) * cos_theta / (p_brdf)

        if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
            rho_s = mat.Kd
            color = cos_theta * rho_s 
        
        color = f_r * cos_theta / p_brdf

        return color
    
    @ti.func
    def shade_ray_light_sampling(self, w_i: tm.vec3, hit_data: HitData, mat: Material, light_triangle_id) -> tm.vec3:

        
        # hit point in this case is the point where our ray hit the light
        color = tm.vec3(0.0)

        light_dist = hit_data.distance
        light_normal = hit_data.normal.normalized()
        light_cos_theta = max(light_normal.dot(w_i), 0)
        light_luminance = mat.Ke
        if light_dist != 0.0 and light_dist > 0.2:
          color = light_luminance * light_cos_theta / (light_dist**2)

        return color
    

    @ti.kernel
    def render(self):
        self.iter_counter[None] += 1.0
        for x,y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x,y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x,y] += (color - self.canvas[x,y])/self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)
        
        if self.shading_mode[None] == int(self.ShadingMode.IMPLICIT):
            color = self.shade_implicit(ray)
        elif self.shading_mode[None] == int(self.ShadingMode.EXPLICIT):
            color = self.shade_explicit(ray)

        return color

    @ti.func
    def shade_implicit(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)

        # TODO A4: Implement Implicit Path Tracing

        # Stopping conditions:
        # - Maximum number of bounces
        # - The object that is hit is a light source
        # - The Ray exits the scene

        # Throughput: this is the accumulated filtering of light as we travel along our path
        # Throughput is multipicative of the color values
        # Light: Light is additive, we add the light we accumulate along the path
        # after termination, our resulting color is the product of the throughput and the light

        T = tm.vec3(1.0)
        L = tm.vec3(0.0)
        # ray is the primary ray
        # cur ray is the current ray we are tracing in our path
        in_ray = ray
        cur_bounce = 0.
        env_map = self.scene_data.environment
        while cur_bounce < self.max_bounces[None]:
            
            # get hit data
            hit_data = self.scene_data.ray_intersector.query_ray(in_ray)
            p = hit_data.distance * in_ray.direction + in_ray.origin
            mat = self.scene_data.material_library.materials[hit_data.material_id]
            rho_s = mat.Kd # specular
            rho_d = mat.Kd # albedo / diffuse
            rho_e = mat.Ke # light emission
            luminosity = rho_e.norm()

            # if hit_data.is_backfacing:
            #     t = tm.vec3(1.0, 0.0, 0.0)
            #     T *= t
            #     break


            # get light emission
            # two cases,
            # if we hit a light source, we use the ke , and we stop the path
            # if we dont hit anything we use the environment map and stop
            if not hit_data.is_hit:
                L_env = env_map.query_ray(ray)
                L += T * L_env
                break
            
            if luminosity > 0.0:
                L += T * rho_e
                break
            
            # sample the direction
            w_i = -in_ray.direction.normalized() # incoming direction
            w_o = self.BRDF.sample_direction(mat, w_i, hit_data.normal) # outgoing direction
            # w_o = self.UNIFORM.sample_direction()
            
            t = self.shade_ray_brdf(w_i, w_o, hit_data, mat) # get throughput at this point of intersection
            T *= t
            out_ray = Ray(p + self.RAY_OFFSET * w_o, w_o.normalized())

            # Update ray
            in_ray = out_ray
            cur_bounce += 1
        

        # TODO A4: Implement Specular Caustics Support - ECSE 546 Deliverable

        color = L
        return color
    
    @ti.func
    def shade_explicit(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)

        T = tm.vec3(1.0)
        L = tm.vec3(0.0)
        # ray is the primary ray
        # cur ray is the current ray we are tracing in our path
        in_ray = ray
        cur_ray_is_light_ray = False
        cur_light_triangle_id = -1
        cur_bounce = 0
        env_map = self.scene_data.environment
        while cur_bounce < self.max_bounces[None]:
            
            # get hit data
            hit_data = self.scene_data.ray_intersector.query_ray(in_ray)
            p = hit_data.distance * in_ray.direction + in_ray.origin
            mat = self.scene_data.material_library.materials[hit_data.material_id]
            rho_s = mat.Kd # specular
            rho_d = mat.Kd # albedo / diffuse
            rho_e = mat.Ke # light emission
            luminosity = rho_e.norm()


            # get light emission
            # two cases,
            # if we hit a light source, we use the ke , and we stop the path
            # if we dont hit anything we use the environment map and stop
            if not hit_data.is_hit:
                L_env = env_map.query_ray(ray)
                L += T * L_env
                break
            
            w_i = -in_ray.direction.normalized()

            if cur_ray_is_light_ray:
                # if we hit the correct light, we add the emission
                if hit_data.triangle_id == cur_light_triangle_id:
                  L += T * self.shade_ray_light_sampling(w_i, hit_data, mat, cur_light_triangle_id)
                # we stop either way if this is a light ray
                break
            
            # if we are not a light ray, and we hit a light source, stop the path
            if luminosity > 0.0:
                if cur_bounce == 0:
                    L += T * rho_e
                break
            
            q = self.rr_termination_probabilty[None]

            # throughput_p = T.norm()
            # q = 1.0 - throughput_p

            # perform russian roulette
            if ti.random() < q:
                break
            
            # sample light randomly 50 50
            sample_light = ti.random() < 0.5
            # sample_light = True

            w_o = hit_data.normal.normalized() # initialize w_o to the normal

            t = tm.vec3(1.0)

            # 

            if sample_light:
                light_direction, light_triangle_id  = self.MLS.sample_direction(p)
                w_o = light_direction.normalized()
                cur_ray_is_light_ray = True
                cur_light_triangle_id = light_triangle_id
                f_r = self.phong_brdf(hit_data, w_i, w_o)
                cos_theta = max(w_o.dot(hit_data.normal), 0)  
                p_r = self.MLS.evaluate_probability()
                t = f_r * cos_theta / p_r
            else:
                while True:
                  w_o = self.BRDF.sample_direction(mat, w_i, hit_data.normal) # outgoing direction
                  test_ray = Ray(p + self.RAY_OFFSET * w_o, w_o.normalized())
                  test_hit_data = self.scene_data.ray_intersector.query_ray(test_ray)
                  mat_test = self.scene_data.material_library.materials[test_hit_data.material_id]
                  # break only if we found a ray not in the light cone
                  if mat_test.Ke.norm() <= 0.0:
                    break
                      
                cur_ray_is_light_ray = False
                cur_light_triangle_id = -1
                t = self.shade_ray_brdf(w_i, w_o, hit_data, mat)

            normal_color = hit_data.normal.normalized()
            normal_color = (normal_color + tm.vec3(1.0)) * 0.5
            T *= t
                   
            # get throughput at this point of intersection
            out_ray = Ray(p + self.RAY_OFFSET * w_o, w_o.normalized())
            in_ray = out_ray
            cur_bounce += 1
        


        # TODO A4: Implement Explicit Path Tracing
        # TODO A4: Implement Russian Roulette Support
        
        color = L
        return color

