import taichi as ti
import taichi.math as tm


@ti.dataclass
class Material:
    """
    Uses notation from the mtl file format:
    https://paulbourke.net/dataformats/mtl/
    """

    Kd: tm.vec3 # Diffuse color
    Ka: tm.vec3 # Ambient color
    Ks: tm.vec3 # Specular color
    Ke: tm.vec3 # Emmissive color

    Ns: float # Specular coefficient
    Ni: float # Optical density or index of refraction
    d: float # Opacity or "dissolve"

    '''
    manually added: 546 only
    '''

    alpha_x: float # x roughness
    alpha_y: float # y roughness
    F0: tm.vec3 # Fresnel term

    def print(self):
        print("Kd: ", self.Kd)
        print("Ka: ", self.Ka)
        print("Ks: ", self.Ks)
        print("Ke: ", self.Ke)
        print("Ns: ", self.Ns)
        print("Ni: ", self.Ni)
        print("d: ",  self.d)


@ti.data_oriented
class MaterialLibrary:
    def __init__(self, material_names_to_id: dict, materials: dict) -> None:
        assert len(materials.keys()) == len(material_names_to_id.keys()), "The number of materials and number of material names do not match!"

        self.material_names_to_id = material_names_to_id
        self.n_materials = len(materials.keys())
        self.materials = Material.field(shape=(self.n_materials))

        # Populate material field with correct indexing
        for material_name in materials.keys():
            material_id = material_names_to_id[material_name]
            self.materials[material_id] = materials[material_name]