from typing import List
from enum import Enum
from importlib import resources

import taichi as ti
import taichi.math as tm
import numpy as np

from .scene_data import SceneData
from .geometry import Geometry
from .materials import Material, MaterialLibrary
from .environment import Environment
from .ray_intersector import BruteForceRayIntersector

from . import scene_data_dir


class SceneName(Enum):
    CUBE = 1
    SPHERE = 2
    TORUS = 3
    MONKEY = 4
    CORNELL_BOX = 5
    BUNNY = 6
    VEACH = 7
    SPECULAR_SPHERES = 8
    BREAKFAST_ROOM = 9
    CORNELL_BOX_CAUSTIC_SPHERE = 10


class EnvironmentName(Enum):
    SHANGHAI = 1
    STUDIO = 2
    FIELD = 3
    WHITE = 4
    BLACK = 5
    GREY = 6


def load_scene_data(
    scene_name: SceneName,
    environment_name: EnvironmentName = EnvironmentName.BLACK,
    ) -> SceneData:
    match scene_name:
        case SceneName.CUBE:
            obj_file = "cube.obj"
        case SceneName.SPHERE:
            obj_file = "sphere.obj"
        case SceneName.TORUS:
            obj_file = "torus.obj"
        case SceneName.MONKEY:
            obj_file = "monkey.obj"
        case SceneName.CORNELL_BOX:
            obj_file = "cornell_box.obj"
        case SceneName.BUNNY:
            obj_file = "bunny.obj"
        case SceneName.VEACH:
            obj_file = "veach_wall.obj"
        case SceneName.SPECULAR_SPHERES:
            obj_file = "specular_spheres.obj"
        case SceneName.BREAKFAST_ROOM:
            obj_file = "breakfast_room.obj"
        case SceneName.CORNELL_BOX_CAUSTIC_SPHERE:
            obj_file = "cornell_box_caustic_sphere.obj"

    # Load geometry
    obj_file_path = resources.files(scene_data_dir) / obj_file
    mtl_file, material_names_to_id, geometry = load_geometry(obj_file_path)

    # Build acceleration structure
    ray_intersector = BruteForceRayIntersector(geometry)

    # Load materials
    mtl_file_path = resources.files(scene_data_dir) / mtl_file
    material_library = load_materials(mtl_file_path, material_names_to_id)

    # Load Environment
    match environment_name:
        case EnvironmentName.SHANGHAI:
            env_file = "shanghai.hdr"
            env_file_path = str(resources.files(scene_data_dir) / env_file)
            environment = load_environment(env_file_path)
        case EnvironmentName.STUDIO:
            env_file = "studio.hdr"
            env_file_path = str(resources.files(scene_data_dir) / env_file)
            environment = load_environment(env_file_path)
        case EnvironmentName.FIELD:
            env_file = "field.hdr"
            env_file_path = str(resources.files(scene_data_dir) / env_file)
            environment = load_environment(env_file_path)
        case EnvironmentName.BLACK:
            environment = Environment(np.zeros((100, 100, 3), dtype=np.float32))
        case EnvironmentName.WHITE:
            environment = Environment(np.ones((100, 100, 3), dtype=np.float32))
        case EnvironmentName.GREY:
            environment = Environment(np.full((100, 100, 3), 0.12, dtype=np.float32))

    return SceneData(
        geometry=geometry,
        material_library=material_library,
        environment=environment,
        ray_intersector=ray_intersector,
        )


def load_geometry(obj_file_path: str) -> Geometry:
    """
    Supports a limited subset of the obj file format that is required for the scenes used in this project.
    https://paulbourke.net/dataformats/obj/
    """

    # Geometery elements
    vertices = []
    normals = []
    texture_coords = []

    # Per-triangle data
    triangle_vertex_ids = []
    triangle_normal_ids = []
    triangle_texture_coord_ids = []
    triangle_material_names = []

    # Tags and files
    mtl_file = None
    current_mtl = None

    # Parse Obj File
    with open(obj_file_path) as file:
        for line in file:
            line = line.rstrip().split(" ")

            match line[0]:

                # Material info
                case "mtllib":
                    mtl_file = line[1]

                case "usemtl":
                    current_mtl = line[1]

                # Parse vertices (x,y,z)
                case "v":
                    vertices.append([line[1], line[2], line[3]])

                # Parse normals (x,y,z)
                case "vn":
                    normals.append([line[1], line[2], line[3]])

                # Parse texture coordinates (u,v)
                case "vt":
                    texture_coords.append([line[1], line[2]])

                # Parse faces
                case "f":
                    if len(line) != 4:
                        raise Exception("This mesh contains non-triangular faces.")

                    triangle_material_names.append(current_mtl)
                    v1 = line[1].split("/")
                    v2 = line[2].split("/")
                    v3 = line[3].split("/")

                    # vertex ids
                    triangle_vertex_ids.append([v1[0], v2[0], v3[0]])

                    # texture coord ids (if available)
                    if (len(v1) >= 2) and (v1[1] != ""):
                        triangle_texture_coord_ids.append([v1[1], v2[1], v3[1]])

                    # normal ids (if available)
                    if (len(v1) == 3) and (v1[2] != ""):
                        triangle_normal_ids.append([v1[2], v2[2], v3[2]])

    # Cast data to np arrays
    vertices = np.array(vertices, dtype=np.float32)
    triangle_vertex_ids = np.array(triangle_vertex_ids, dtype=np.int32)

    normals = np.array(normals, dtype=np.float32)
    triangle_normal_ids = np.array(triangle_normal_ids, dtype=np.int32)

    # Convert the material names into ids
    # Retain a dictionary of the mapping
    material_names = list(set(triangle_material_names))
    material_names_to_id = dict(zip(material_names, range(len(material_names))))
    triangle_material_ids = [material_names_to_id[x] for x in triangle_material_names]
    triangle_material_ids = np.array(triangle_material_ids, dtype=np.int32)

    # Optional data
    texture_coords = (
        np.array(texture_coords, dtype=np.float32) if len(texture_coords) else None
    )
    triangle_texture_coord_ids = (
        np.array(triangle_texture_coord_ids, dtype=np.int32)
        if len(triangle_texture_coord_ids)
        else None
    )

    g = Geometry(
        vertices=vertices,
        triangle_vertex_ids=triangle_vertex_ids,
        normals=normals,
        triangle_normal_ids=triangle_normal_ids,
        triangle_material_ids=triangle_material_ids,
        texture_coords=texture_coords,
        triangle_texture_coord_ids=triangle_texture_coord_ids,
    )

    return mtl_file, material_names_to_id, g


def load_materials(mtl_file_path: str, material_names_to_id: dict) -> MaterialLibrary:
    """
    Supports a limited subset of the mtl file format that is required for the scenes used in this project.
    https://paulbourke.net/dataformats/mtl/
    """

    materials = {}
    active_mtl = None

    # Parse the mtl file
    with open(mtl_file_path) as file:
        for line in file:
            line = line.strip().split(" ")
            line = [x for x in line if x != ""]
            if not len(line):
                continue

            match line[0]:

                case "newmtl":
                    active_mtl = line[1]
                    materials[active_mtl] = Material()

                case "Kd":
                    materials[active_mtl].Kd = tm.vec3(
                        [
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                        ]
                    )

                case "Ka":
                    materials[active_mtl].Ka = tm.vec3(
                        [
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                        ]
                    )

                case "Ks":
                    materials[active_mtl].Ks = tm.vec3(
                        [
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                        ]
                    )

                case "Ke":
                    materials[active_mtl].Ke = tm.vec3(
                        [
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                        ]
                    )

                case "Ns":
                    materials[active_mtl].Ns = float(line[1])

                case "Ni":
                    materials[active_mtl].Ni = float(line[1])

                case "d":
                    materials[active_mtl].d = float(line[1])

                case "alpha_x":
                    materials[active_mtl].alpha_x = float(line[1])
                
                case "alpha_y":
                    materials[active_mtl].alpha_y = float(line[1])
                
                case "F0":
                    materials[active_mtl].F0 = tm.vec3(
                        [
                            float(line[1]),
                            float(line[2]),
                            float(line[3]),
                        ]
                    )
                


    return MaterialLibrary(material_names_to_id, materials)


def load_environment(env_file_path: str) -> Environment:
    image = ti.tools.imread(env_file_path).astype(np.float32) / 255.0
    return Environment(image=image)