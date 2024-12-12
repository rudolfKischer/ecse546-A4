from typing import List

import taichi as ti
import taichi.math as tm
import numpy as np

from .geometry import Geometry
from .materials import MaterialLibrary, Material

@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)

@ti.data_oriented
class UniformSampler:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction() -> tm.vec3:
        # we want to sample on the unit sphere
        # w = [w_x, w_y, w_z]
        X_1 = ti.random()
        X_2 = ti.random()
        # first we pick a height between -1 and 1
        w_z = 2. * X_1 - 1.
        # we then want to pick an angle on the ring on this level set
        # we pick an angle between 0 and 2pi
        theta = X_2 * 2. * tm.pi
        # now we need to find the radius of the circle on that level set
        # we know that the hypotenuse is eqault to the radius of the sphere, which is 1 because we are on the unit sphere
        # and out opposite, which is our height, is w_z
        # a^2 + b^2 = c^2 =>  a^2 + w_z^2 = 1^2 => a = sqrt(1 - w_z^2)
        # this gives us the radius of the circle on the level set at high w_z
        r = tm.sqrt(1. - w_z**2)
        # we know cos(theta) = adjacent/hypotenuse = w_x/r => w_x = r * cos(theta)
        w_x = r * tm.cos(theta)
        # we know sin(theta) = opposite/hypotenuse = w_y/r => w_y = r * sin(theta)
        w_y = r * tm.sin(theta)
        w = tm.vec3(w_x, w_y, w_z)
        # normalize the vector
        w = w.normalized()
        return w




    @staticmethod
    @ti.func
    def evaluate_probability() -> float:
        return 1. / (4. * tm.pi)

@ti.data_oriented
class BRDF:
    def __init__(self):
        return 

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        w_o = w_o.normalized()
        normal = normal.normalized()
        X_1 = ti.random()
        X_2 = ti.random()

        w_r = reflect(w_o, normal).normalized()

        a = material.Ns

        lobe_term = 1. / (a + 1.)

        w_hat_z = ti.pow(X_1, lobe_term)


        phi = 2. * tm.pi * X_2
        r = ti.sqrt(1. - w_hat_z**2)
        w_hat_x = r * tm.cos(phi)
        w_hat_y = r * tm.sin(phi)

        w_hat = tm.vec3(w_hat_x, w_hat_y, w_hat_z)
        w_hat = w_hat.normalized()

        # now that we have the direction about the canonical frame, we need to align them along the normal
        frame = ortho_frames(normal.normalized())
        if a > 1.:
          frame = ortho_frames(w_r)
        # w_hat = tm.vec3([0., 0., 1.])
        w_i = frame @ w_hat
        return w_i


    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> float: 
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        normal = normal.normalized()
        result = (1. / tm.pi) * max(1e-6, w_o.dot(normal))
        a = material.Ns
        w_r = reflect(w_o, normal).normalized()

          
        if a > 1.:
            wi_dot_wr = max(0.0,w_r.dot(w_i))
            # if wo_dot_wr > (1. - 1e-6):
              # wo_dot_wr = 1. - 1e-6


            lobe_term = tm.pow(wi_dot_wr, a)
            if lobe_term > 1.0:
              lobe_term = 1.0

            if a > 60:
                print(f'alpha: {a}, lobe_term: {lobe_term:.5e} wr_dot_wi: {wi_dot_wr:.5e}')
            rho_s = material.Kd

            result = ((a + 1.) / (2. * tm.pi)) * max(lobe_term,0.0)
        return result

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        normal = normal.normalized()
        result = tm.vec3([0., 0., 0.])
        a = material.Ns
        rho_s = material.Ks
        rho_d = material.Kd
        if a > 1.:
            result = rho_d * (max(0., normal.dot(w_o)))
        else:
            result = rho_d
        return result
    
    @staticmethod
    @ti.func
    def evaluate_brdf_factor(material: Material, w_o: tm.vec3, w_i: tm.vec3, normal: tm.vec3) -> tm.vec3:
        pass

@ti.data_oriented
class MicrofacetBRDF:
    def __init__(self):
        pass

    @staticmethod
    @ti.func
    def sample_direction(material: Material, w_o: tm.vec3, normal: tm.vec3) -> tm.vec3:
        a_x = material.alpha_x
        a_y = material.alpha_y
        F0 = material.F0
        w_o = w_o.normalized()
        normal = normal.normalized()

        # move everything into the normal frame
        frame = ortho_frames(normal)
        w_o = (frame.transpose() @ w_o).normalized()
        w_h = tm.vec3([a_x * w_o[0], a_y * w_o[1], w_o[2]]).normalized()
        # flip the vector if z component is negative
        if w_h[2] < 0:
            w_h = -w_h
        
        # form orthonormal basis
        machine_epsilon = 1e-6
        t = tm.vec3([1, 0, 0])
        if abs(w_h[2]) <= 1 - machine_epsilon:
            t = tm.vec3([0, 0, 1]).cross(w_h).normalized()
        
        b = w_h.cross(t).normalized()

        # sample the microfacet normal
        X1 = ti.random()
        X2 = ti.random()
        r = ti.sqrt(X1)
        phi = 2. * tm.pi * X2
        p = tm.vec3([r * tm.cos(phi), r * tm.sin(phi), 0.])

        # we now have a point sampled on unit disk

        # now we want to warp the sampled point to adjust for the reflection vector
        p_x = p[0]
        h = ti.sqrt(1. - p_x**2) 

        mix1 = h
        mix2 = p[1] 
        p_y = tm.mix(mix1, mix2, (1. + w_h[2]) / 2.)
        p_z = ti.sqrt(max(0., 1. - p_x**2 - p_y**2))

        n_h = p_x * t + p_y * b + p_z * w_h
        n_h = n_h.normalized()

        w_m = tm.vec3(a_x * n_h[0], a_y * n_h[1], max(machine_epsilon, n_h[2])).normalized()

        # now we need to rotate the microfacet normal back to the world frame
        # we use the inverse of the frame matrix, which is the transpose
        w_m = (frame @ w_m).normalized()

        return w_m
    
    @staticmethod
    @ti.func
    def Lambda(material: Material, w: tm.vec3) -> float:
        # given w we can get theta and phi
        # assuming w is in normal frame
        # cos(phi) = x / r = x / sin(theta)
        # sin(phi) = y / r = y / sin(theta)
        # sin(theta)^2 + cos(theta)^2 = 1
        # sin(theta) = sqrt(1 - cos(theta)^2)
        # cos(theta) = w[z]
        a_x = material.alpha_x
        a_y = material.alpha_y
        theta = tm.acos(w[2])
        phi = tm.atan2(w[1], w[0])
        alpha = tm.sqrt(a_x**2 * tm.cos(phi)**2 + a_y**2 * tm.sin(phi)**2)
        lambda_w = (tm.sqrt(1 + alpha**2 * tm.tan(theta)**2) - 1) / 2
        return lambda_w
    
    @staticmethod
    @ti.func
    def G_1(material: Material, w: tm.vec3, normal: tm.vec3) -> float:
        w = w.normalized()
        frame = ortho_frames(normal)
        w = (frame.transpose() @ w).normalized()
        lambda_w = MicrofacetBRDF.Lambda(material, w)
        return 1. / (1. + lambda_w)
    
    @staticmethod
    @ti.func
    def G(material: Material, w_i: tm.vec3, w_o: tm.vec3, normal: tm.vec3) -> float:
        # convert w_i and w_o to normal frame
        w_i = w_i.normalized()
        w_o = w_o.normalized()
        normal = normal.normalized()
        frame = ortho_frames(normal)
        w_i = (frame.transpose() @ w_i).normalized()
        w_o = (frame.transpose() @ w_o).normalized()

        lambda_w_o = MicrofacetBRDF.Lambda(material, w_o)
        lambda_w_i = MicrofacetBRDF.Lambda(material, w_i)
        return 1. / (1. + lambda_w_o + lambda_w_i)
    
    @staticmethod
    @ti.func
    def D_wm(material: Material, w_m: tm.vec3, normal: tm.vec3) -> float:
        normal = normal.normalized()
        w_m = w_m.normalized()
        a_x = material.alpha_x
        a_y = material.alpha_y
        frame = ortho_frames(normal)
        w_m = (frame.transpose() @ w_m).normalized()
        w_m = w_m.normalized()

        theta_m = tm.acos(w_m[2])
        # if the absolute value falls below machine epsilon, we set it to machine epsilon
        # make sure to maintain the sign we want w_m[0]
        phi_m = tm.atan2(w_m[1], tm.sign(w_m[0]) * max(1e-6, abs(w_m[0])))
        denominator = tm.pi * a_x * a_y * tm.cos(theta_m)**4
        squared_term = (1 + tm.tan(theta_m)**2 * ((tm.cos(phi_m) / a_x)**2 + (tm.sin(phi_m) / a_y)**2))
        denominator = denominator * squared_term**2

        return 1. / max(denominator, 1e-6)

    @staticmethod
    @ti.func
    def D(material: Material, w: tm.vec3, w_m: tm.vec3, normal: tm.vec3) -> float:
        # G1(w) * D(w_m) * max(0, w . w_m) / w . n
        w = w.normalized()
        w_m = w_m.normalized()
        normal = normal.normalized()
        G_1_w = MicrofacetBRDF.G_1(material, w, normal)
        D_wm = MicrofacetBRDF.D_wm(material, w_m, normal)
        return  D_wm *max(1e-6, w.dot(w_m)) / max(1e-6, w.dot(normal)) * G_1_w
    
    @staticmethod
    @ti.func
    def F(material: Material, w_o: tm.vec3, w_i: tm.vec3) -> float:
        F0 = material.F0
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        w_o_dot_w_i = max(1e-6, w_o.dot(w_i))
        return F0 + (1. - F0) * (1. - w_o_dot_w_i)**5



    @staticmethod
    @ti.func
    def evaluate_probability(material: Material, w_o: tm.vec3, w_i: tm.vec3, w_m: tm.vec3, normal: tm.vec3) -> float: 
        
        w_o = w_o.normalized()
        w_i = w_i.normalized()
        w_m = w_m.normalized()
        normal = normal.normalized()
        numerator =  MicrofacetBRDF.D(material, w_o, w_m, normal)
        denominator = 4. * max(w_o.dot(w_m), 1e-6)# * max(w_i.dot(normal), 1e-6)
        return numerator / denominator
        

    @staticmethod
    @ti.func
    def evaluate_brdf(material: Material, w_o: tm.vec3, w_i: tm.vec3, w_m: tm.vec3, normal: tm.vec3) -> tm.vec3:
        # D(w_m) F(w_o, w_m) G(w_i, w_o, w_m) / (4 w_i . n w_o . n)
        D_wm = MicrofacetBRDF.D_wm(material, w_m, normal)
        F_wm = MicrofacetBRDF.F(material, w_o, w_m)
        G_w = MicrofacetBRDF.G(material, w_i, w_o, normal)
        w_i = w_i.normalized()
        w_o = w_o.normalized()
        normal = normal.normalized()
        numerator =  F_wm * G_w * D_wm
        denominator = 4. * max(w_o.dot(normal), 1e-6) * max(w_i.dot(normal), 1e-6)
        return numerator / (denominator)


@ti.data_oriented
class MeshLightSampler:
    def __init__(self, geometry: Geometry, material_library: MaterialLibrary):
        self.geometry = geometry
        self.material_library = material_library

        # Find all of the emissive triangles
        emissive_triangle_ids = self.get_emissive_triangle_indices()
        if len(emissive_triangle_ids) == 0:
            self.has_emissive_triangles = False
        else:
            self.has_emissive_triangles = True
            self.n_emissive_triangles = len(emissive_triangle_ids)
            emissive_triangle_ids = np.array(emissive_triangle_ids, dtype=int)
            self.emissive_triangle_ids = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=int)
            self.emissive_triangle_ids.from_numpy(emissive_triangle_ids)

        # Setup for importance sampling
        if self.has_emissive_triangles:
            # Data Fields
            self.emissive_triangle_areas = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.cdf = ti.field(shape=(emissive_triangle_ids.shape[0]), dtype=float)
            self.total_emissive_area = ti.field(shape=(), dtype=float)

            # Compute
            self.compute_emissive_triangle_areas()
            self.compute_cdf()


    def get_emissive_triangle_indices(self) -> List[int]:
        # Iterate over each triangle, and check for emissivity 
        emissive_triangle_ids = []
        for triangle_id in range(1, self.geometry.n_triangles + 1):
            material_id = self.geometry.triangle_material_ids[triangle_id-1]
            emissivity = self.material_library.materials[material_id].Ke
            if emissivity.norm() > 0:
                emissive_triangle_ids.append(triangle_id)

        return emissive_triangle_ids


    @ti.kernel
    def compute_emissive_triangle_areas(self):
        for i in range(self.n_emissive_triangles):
            triangle_id = self.emissive_triangle_ids[i]
            vert_ids = self.geometry.triangle_vertex_ids[triangle_id-1] - 1  # Vertices are indexed from 1
            v0 = self.geometry.vertices[vert_ids[0]]
            v1 = self.geometry.vertices[vert_ids[1]]
            v2 = self.geometry.vertices[vert_ids[2]]

            triangle_area = self.compute_triangle_area(v0, v1, v2)
            self.emissive_triangle_areas[i] = triangle_area
            self.total_emissive_area[None] += triangle_area
        

    @ti.func
    def compute_triangle_area(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> float:
        A = v0
        B = v1
        C = v2

        AB = B - A
        AC = C - A

        area = 0.5 * AB.cross(AC).norm()

        return area


    @ti.kernel
    def compute_cdf(self):

        sum_val = 0.0
        ti.loop_config(serialize=True)
        for i in range(self.n_emissive_triangles):
            sum_val += self.emissive_triangle_areas[i]
            self.cdf[i] = sum_val / self.total_emissive_area[None]

        ti.loop_config(serialize=False)




    @ti.func
    def sample_emissive_triangle(self) -> int:
        result = 0
        X = ti.random()
        for i in range(self.n_emissive_triangles):
            if X > self.cdf[i]:
                result = i

        # TODO: convert to binary search
        
        # l = 0
        # r = self.n_emissive_triangles - 1
        # while l < r:
        #     m = (l + r) // 2
        #     if self.cdf[m] < X:
        #         l = m + 1
        #     else:
        #         r = m
        # result = r

        #result = int(self.n_emissive_triangles * X)

        return result

    @ti.func
    def sample_bary_point(self, v0: tm.vec3, v1: tm.vec3, v2: tm.vec3) -> tm.vec3:

        X1 = ti.random()
        X2 = ti.random()

        if X1 > X2:
            # swap
            X1, X2 = X2, X1

        b0 = X1 * 0.5 
        b1 = X2 - b0
        b2 = 1.0 - b0 - b1

        y = b0 * v0 + b1 * v1 + b2 * v2
        return y

    @ti.func
    def evaluate_probability(self) -> float:
        
        # because we are sampling uniformely over the area of the triangles
        # and we are weighting probability of the triangle by their area, 
        # the probability of sampling a point on a triangle is 1 / total_emissive_area

        # p_triangle = triangle_area / total_triangle_area
        # p_point = 1 / triangle_area
        # p = p_point * p_triangle = triangle_area / total_triangle_area * 1 / triangle_area = 1 / total_triangle_area

        p = 1.0 / self.total_emissive_area[None]

        return p


    @ti.func
    def sample_mesh_lights(self, hit_point: tm.vec3):
        sampled_light_triangle_emissive_idx = self.sample_emissive_triangle()
        sampled_light_triangle_idx = self.emissive_triangle_ids[sampled_light_triangle_emissive_idx]

        # Grab Vertices
        vert_ids = self.geometry.triangle_vertex_ids[sampled_light_triangle_idx-1] - 1  # Vertices are indexed from 1
        
        v0 = self.geometry.vertices[vert_ids[0]]
        v1 = self.geometry.vertices[vert_ids[1]]
        v2 = self.geometry.vertices[vert_ids[2]]
        sample_point = self.sample_bary_point(v0, v1, v2)
        light_direction = (sample_point - hit_point).normalized()
        
        # placeholder
        return light_direction, sampled_light_triangle_idx

    @ti.func
    def sample_direction(self, hit_point: tm.vec3) ->tm.vec3:

        light_direction, sampled_triangle = self.sample_mesh_lights(hit_point)
        light_direction = light_direction.normalized()

        return light_direction, sampled_triangle



@ti.func
def ortho_frames(v_z: tm.vec3) -> tm.mat3:
    # random_vec = tm.vec3([0, 1 ,0 ]).normalized()
    random_vec = tm.vec3([ti.random(), ti.random(), ti.random()]).normalized()
    v_x = v_z.cross(random_vec).normalized()
    v_y = v_x.cross(v_z).normalized()
    ortho_frames = tm.mat3(v_x, v_y, v_z)
    return ortho_frames.transpose() 


@ti.func
def reflect(ray_direction:tm.vec3, normal: tm.vec3) -> tm.vec3:
    return (2 * ray_direction.dot(normal) * normal - ray_direction).normalized()