import bpy
import os
import json
import numpy as np
import random
from mathutils import Vector
import sys

PROCEDURAL_TEXTURES = ["Voronoi Texture", "Wave Texture", "Musgrave Texture", "Noise Texture"]

def setup_scene(resolution):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(confirm=False)
    
    bpy.ops.mesh.primitive_plane_add(size=2.0, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.active_object
    
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 2))
    camera = bpy.context.active_object
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera
    
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
        
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    
    for node in world_nodes:
        world_nodes.remove(node)
    
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs[0].default_value = (1, 1, 1, 1)
    background.inputs[1].default_value = 1.0
    
    output = world_nodes.new(type='ShaderNodeOutputWorld')
    
    world.node_tree.links.new(background.outputs[0], output.inputs[0])
    
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    
    try:
        bpy.context.scene.cycles.device = 'GPU'
    except:
        bpy.context.scene.cycles.device = 'CPU'
    
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    return plane

def create_noise_texture():
    config = {}
    
    scale = random.uniform(5.0, 10.0)     
    detail = random.uniform(5.0, 10.0)    
    roughness = random.uniform(0.3, 1.0)  
    distortion = random.uniform(1.0, 8.0) 
    
    config["input_values"] = {
        "W": 0.0,
        "Scale": scale,
        "Detail": detail,
        "Roughness": roughness,
        "Distortion": distortion
    }
    
    config["texture_type"] = "Noise Texture"
    config["output_name"] = "Fac"
    config["input_node"] = "Mapping"
    config["output_node"] = "ColorRamp"
    
    return config

def create_voronoi_texture():
    config = {}
    

    feature = random.choice(['F1', 'F2'])
    distance = random.choice(['EUCLIDEAN', 'MANHATTAN', 'CHEBYCHEV', 'MINKOWSKI'])
    
    scale = random.uniform(8.0, 15.0)  
    
    config["feature"] = feature
    config["distance"] = distance
    config["input_values"] = {
        "W": 0.0,
        "Scale": scale,
        "Smoothness": random.uniform(0.0, 1.0), 
        "Exponent": random.uniform(0.5, 2.0) if distance == 'MINKOWSKI' else 0.5,
        "Randomness": random.uniform(0.7, 1.0) 
    }
    
    config["texture_type"] = "Voronoi Texture"
    config["output_name"] = "Distance"
    config["input_node"] = "Mapping"
    config["output_node"] = "ColorRamp"
    
    return config

def create_wave_texture():
    config = {}
    
    config["wave_type"] = random.choice(['BANDS', 'RINGS'])
    config["wave_profile"] = random.choice(['SIN', 'SAW', 'TRI'])
    config["bands_direction"] = random.choice(['X', 'Y', 'Z'])
    config["rings_direction"] = random.choice(['X', 'Y', 'Z', 'SPHERICAL'])
    
    scale = random.uniform(1.0, 8.0)
    distortion = random.uniform(5.0, 25.0) 
    
    config["input_values"] = {
        "Scale": scale,
        "Distortion": distortion,
        "Detail": random.uniform(0.0, 2.0),
        "Detail Scale": random.uniform(1.0, 4.0),
        "Detail Roughness": random.uniform(0.5, 1.0),
        "Phase Offset": random.uniform(0.0, 1.0)
    }
    
    config["texture_type"] = "Wave Texture"
    config["output_name"] = "Color"
    config["input_node"] = "Mapping"
    config["output_node"] = "ColorRamp"
    
    return config

def create_musgrave_texture():
    config = {}
    
    config["musgrave_type"] = random.choice(['FBM', 'MULTIFRACTAL', 'HYBRID_MULTIFRACTAL', 'RIDGED_MULTIFRACTAL'])
    
    scale = random.uniform(6.0, 15.0)
    detail = random.uniform(4.0, 8.0)
    dimension = random.uniform(0.0, 2.0)
    
    config["input_values"] = {
        "W": 0.0,
        "Scale": scale,
        "Detail": detail,
        "Dimension": dimension,
        "Lacunarity": random.uniform(1.5, 4.0),
        "Offset": random.uniform(0.0, 1.0), 
    }
    
    config["texture_type"] = "Musgrave Texture"
    config["output_name"] = "Fac"
    config["input_node"] = "Mapping"
    config["output_node"] = "ColorRamp"
    
    return config

def create_texture_config(texture_type):
    if texture_type == "Noise Texture":
        return create_noise_texture()
    elif texture_type == "Voronoi Texture":
        return create_voronoi_texture()
    elif texture_type == "Wave Texture":
        return create_wave_texture()
    elif texture_type == "Musgrave Texture":
        return create_musgrave_texture()
    else:
        return create_noise_texture() 

def random_color():

    return (random.random(), random.random(), random.random(), 1.0)

def create_material(texture_config, obj, material_name="Material", use_colors=False):

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    

    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    mapping_node = nodes.new(type='ShaderNodeMapping')
    coord_node = nodes.new(type='ShaderNodeTexCoord')
    

    mapping_node.inputs['Location'].default_value = (
        random.uniform(-2, 2),
        random.uniform(-2, 2),
        random.uniform(-2, 2)
    )
    mapping_node.inputs['Rotation'].default_value = (
        random.uniform(0, 6.28),  
        random.uniform(0, 6.28),
        random.uniform(0, 6.28)
    )
    mapping_node.inputs['Scale'].default_value = (
        random.uniform(0.5, 2.0),
        random.uniform(0.5, 2.0),
        random.uniform(0.5, 2.0)
    )
    
    output_node.location = (600, 0)
    emission_node.location = (400, 0)
    color_ramp.location = (200, 0)
    mapping_node.location = (-200, 0)
    coord_node.location = (-400, 0)
    
    texture_type = texture_config["texture_type"]
    if texture_type == "Noise Texture":
        texture = nodes.new(type='ShaderNodeTexNoise')
    elif texture_type == "Voronoi Texture":
        texture = nodes.new(type='ShaderNodeTexVoronoi')
        texture.feature = texture_config["feature"]
        texture.distance = texture_config["distance"]
    elif texture_type == "Wave Texture":
        texture = nodes.new(type='ShaderNodeTexWave')
        texture.wave_type = texture_config["wave_type"]
        texture.wave_profile = texture_config["wave_profile"]
        texture.rings_direction = texture_config["rings_direction"]
        if "bands_direction" in texture_config:
            try:
                texture.bands_direction = texture_config["bands_direction"]
            except:
                pass  
    elif texture_type == "Musgrave Texture":
        texture = nodes.new(type='ShaderNodeTexMusgrave')
        texture.musgrave_type = texture_config["musgrave_type"]
    
    texture.location = (0, 0)
    

    for param, value in texture_config["input_values"].items():
        if param in texture.inputs:
            texture.inputs[param].default_value = value
    

    links = mat.node_tree.links
    links.new(emission_node.outputs[0], output_node.inputs[0])
    links.new(color_ramp.outputs[0], emission_node.inputs[0])
    links.new(texture.outputs[0], color_ramp.inputs[0])
    links.new(mapping_node.outputs[0], texture.inputs[0])
    links.new(coord_node.outputs["Object"], mapping_node.inputs[0])
    

    for _ in range(len(color_ramp.color_ramp.elements) - 1):
        color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    
    if not use_colors:

        color_ramp.color_ramp.elements[0].position = 0.2  
        color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
        

        element = color_ramp.color_ramp.elements.new(0.8) 
        element.color = (1, 1, 1, 1)
        

        if random.random() < 0.5:
            gray_val = random.uniform(0.3, 0.7)
            mid_element = color_ramp.color_ramp.elements.new(0.5)
            mid_element.color = (gray_val, gray_val, gray_val, 1)
        

        color_ramp.color_ramp.interpolation = random.choice(['CONSTANT', 'B_SPLINE', 'LINEAR'])
    else:
        num_colors = random.randint(3, 5)
        

        position_range = 1.0 / (num_colors - 1)
        
        color_ramp.color_ramp.elements[0].position = 0.0
        color_ramp.color_ramp.elements[0].color = random_color()
        

        for i in range(1, num_colors):

            position = i * position_range + random.uniform(-0.05, 0.05)
            position = max(0.01, min(0.99, position))  
            
            element = color_ramp.color_ramp.elements.new(position)
            element.color = random_color()
        

        color_ramp.color_ramp.interpolation = random.choice(['EASE', 'CARDINAL', 'LINEAR', 'B_SPLINE'])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return mat

def render_texture(config, output_path, resolution=512, use_colors=False):

    plane = setup_scene(resolution)
    

    create_material(config, plane, use_colors=use_colors)
    

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def main():

    output_dir = "texture_exports"

    
    os.makedirs(output_dir, exist_ok=True)

    

    textures_per_category = 100
    if len(sys.argv) > 1:
        try:
            textures_per_category = int(sys.argv[1])
        except:
            pass
    

    texture_index = 0
    
    for texture_type in PROCEDURAL_TEXTURES:
        print(f"\nGenerating {textures_per_category} {texture_type} textures...")

        
        for i in range(textures_per_category):

            foreground_config = create_texture_config(texture_type)
            background_config = create_texture_config(texture_type)
            

            if "Scale" in background_config["input_values"]:
                background_config["input_values"]["Scale"] = foreground_config["input_values"]["Scale"] * 4.0
            
            config = {
                "foreground": {texture_type: foreground_config},
                "background": {texture_type: background_config}
            }
            
                

            use_colors = (i % 2 == 0)  
            color_text = "colored" if use_colors else "black and white"
            
            print(f"  Generated {texture_type} configuration {i+1}/{textures_per_category} ({color_text})")
            
            
            fg_path = os.path.join(output_dir, f"texture_{texture_index:03d}.png")
            render_texture(foreground_config, fg_path, use_colors=use_colors)
            print(f"  Rendered foreground texture to {fg_path}")
            

            bg_path = os.path.join(output_dir, f"texture_{texture_index+1:03d}.png")
            render_texture(background_config, bg_path, use_colors=use_colors)
            print(f"  Rendered background texture to {bg_path}")
            
            texture_index += 2
        
    print(f"\nCompleted: Generated {texture_index} textures across {len(PROCEDURAL_TEXTURES)} categories")

if __name__ == "__main__":
    main()