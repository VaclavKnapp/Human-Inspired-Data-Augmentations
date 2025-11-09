import os
import sys
import argparse
import pickle
import bpy
import numpy as np
from mathutils import Vector

def generate_object(params, clear=1):
    print('STARTING')
    if clear:
        delete_everything()
    # Generate object using the shape_generator addon with specified parameters
    bpy.ops.mesh.shape_generator(
        random_seed=params['seed'],
        mirror_x=0,
        mirror_y=0,
        mirror_z=0,
        favour_vec=params['axis_preferences'],
        amount=params['n_extrusions'],
        is_subsurf=(params['n_subdivisions'] > 0),
        subsurf_subdivisions=params['n_subdivisions'],
        big_shape_num=params['n_bigshapes'],
        medium_shape_num=params['n_medshapes'],
        medium_random_scatter_seed=params['med_locseed'],
        big_random_scatter_seed=params['big_locseed'],
        big_random_seed=params['big_shapeseed'],
        medium_random_seed=params['med_shapeseed'],
        # Additional parameters can be added here if needed.
    )
    print('OVER')

    for obj in bpy.data.objects:
        obj.select_set(False)

def scale_and_center_object(xyz=(0,0,0)):
    """
    Scales the object so that it fits within a unit sphere and centers it at the specified location.
    """
    def bbox(ob):
        return (Vector(b) for b in ob.bound_box)
    def bbox_center(ob):
        return sum(bbox(ob), Vector()) / 8
    def mesh_radius(ob):
        o = bbox_center(ob)
        return max((v.co - o).length for v in ob.data.vertices)
    
    # Get the selected object (assumes that one object is selected after joining)
    obj = bpy.context.selected_objects[0]
    scale_factor = 1 / mesh_radius(obj)
    obj.scale *= scale_factor
    obj.location = xyz

def delete_everything():
    # Delete all collections
    for coll in bpy.data.collections:
        bpy.data.collections.remove(coll)
    # Delete all objects
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj)

def select_random_texture(texture_dir):
    """
    Randomly selects a .png texture file from the specified directory.
    
    Args:
        texture_dir (str): Path to the directory containing texture files
        
    Returns:
        str: Full path to the selected texture file or None if no textures found
    """
    if not os.path.exists(texture_dir):
        print(f"Texture directory {texture_dir} does not exist.")
        return None
    
    texture_files = [f for f in os.listdir(texture_dir) if f.lower().endswith('.png')]
    
    if not texture_files:
        print(f"No .png texture files found in {texture_dir}.")
        return None
    
    selected_texture = np.random.choice(texture_files)
    print(f"Selected texture: {selected_texture}")
    return os.path.join(texture_dir, selected_texture)

def apply_texture(obj, texture_path):
    """
    Applies a texture to the specified object.
    
    Args:
        obj: Blender object to apply texture to
        texture_path (str): Path to the texture image file
    """
    # Make sure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Create a new material
    mat = bpy.data.materials.new(name="TextureMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create texture coordinate node
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    
    # Create mapping node
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-600, 0)
    
    # Create image texture node
    tex_image = nodes.new(type='ShaderNodeTexImage')
    tex_image.location = (-400, 0)
    
    # Load image
    tex_image.image = bpy.data.images.load(texture_path)
    
    # Create BSDF shader node
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    
    # Link nodes
    links = mat.node_tree.links
    links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex_image.inputs["Vector"])
    links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Add UV unwrapping - smart UV project
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"Applied texture {os.path.basename(texture_path)} to {obj.name}")

def process_shape(params, output_dir, smoothness_name, shape_count, args):
    """
    Process a single shape with the given parameters, save it, and return the shape name
    """
    # Generate the object
    delete_everything()
    generate_object(params)
    
    # ----- Adjust rotations for generated shapes -----
    collections = bpy.data.collections
    # Get medium shapes (if any)
    med_coll = [coll for coll in collections if 'Medium' in coll.name]
    med_objects = med_coll[0].objects[:] if med_coll else []
    
    # Get big shapes (look for a collection named exactly 'Generated Shape Collection')
    big_coll = [coll for coll in collections if coll.name == 'Generated Shape Collection']
    big_objects = big_coll[0].objects[:] if big_coll else []
    
    # Apply rotations to big shapes (skip the object named 'Generated Shape')
    brotation = params['brotations'][0]
    for obj in big_objects:
        if obj.name != 'Generated Shape':
            obj.rotation_euler.x -= brotation * np.pi
    
    # Apply rotations to medium shapes and adjust their location
    mrotation = params['mrotations'][0]
    for obj in med_objects:
        obj.rotation_euler.x -= mrotation * np.pi
        obj.location.x += 0.3
    
    # ----- Join all objects into one and center/scale the result -----
    for obj in bpy.data.objects:
        obj.select_set(True)
    bpy.ops.object.join()
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    bpy.ops.object.location_clear(clear_delta=False)
    scale_and_center_object()
    
    # ----- Apply texture if texture directory is provided -----
    if args.textures:
        # Get the selected object
        obj = bpy.context.selected_objects[0]
        # Select and apply a random texture
        texture_path = select_random_texture(args.textures)
        if texture_path:
            apply_texture(obj, texture_path)
            # Store texture info
            params['texture'] = os.path.basename(texture_path)
    
    # ----- Save the resulting object -----
    shape_name = f"shape_{smoothness_name}_{shape_count:03d}"
    save_path = os.path.join(output_dir, shape_name + '.blend')
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    
    return shape_name

def get_random_params(n_extrusions):
    """
    Generate random parameters for a shape
    """
    # [np.random.uniform(0, .2), np.random.uniform(.6, 1), np.random.uniform(.6, 1)]
    params = {
        'axis_preferences': [
            np.random.uniform(0.7, 1),
            np.random.uniform(0.7, 1),
            np.random.uniform(0.7, 1)  # 0 - 1
        ],
        'n_bigshapes': 1,
        'n_medshapes': 0,
        'big_shapeseed': np.random.randint(1000),
        'big_locseed': np.random.randint(1000),
        'med_locseed': np.random.randint(1000),
        'med_shapeseed': np.random.randint(1000),
        'n_extrusions': int(n_extrusions),
        'seed': np.random.randint(1000),
        'brotations': [np.random.uniform(-0.5, 0.5)], 
        'mrotations': [np.random.uniform(-0.5, 0.5)],
    }
    return params

def ensure_nested_dict_structure(stimulus_info, extrusion_key, smoothness_key):
    """
    Ensure the nested dictionary structure exists for the given keys
    """
    if extrusion_key not in stimulus_info:
        stimulus_info[extrusion_key] = {}
    if smoothness_key not in stimulus_info[extrusion_key]:
        stimulus_info[extrusion_key][smoothness_key] = {}

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description="Generate random shapes with variable smoothness levels.")
    parser.add_argument("-n_extrusions", type=int, default=None,
                        help="Number of extrusions (default: determined by preview mode)")
    parser.add_argument("-o", "--output", type=str,
                        default=os.path.join(os.getcwd(), "generated_shapes"),
                        help="Output directory (default: <cwd>/generated_shapes)")
    parser.add_argument("-n_shapes", type=int, default=5,
                        help="Number of random shapes to generate per smoothness type (default: 5)")
    parser.add_argument("-s", "--smoothness_levels", type=int, default=2,
                        help="Number of smoothness levels to generate (default: 2, max: 6)")
    parser.add_argument("-t", "--textures", type=str,
                        help="Directory containing texture .png files")
    parser.add_argument("--preview", action="store_true",
                        help="Generate preview set: 3 shapes for each smoothness level (0-5) and n_extrusions (2-9)")
    args = parser.parse_args()
    
    # Validate smoothness levels
    if args.smoothness_levels < 1:
        args.smoothness_levels = 1
    elif args.smoothness_levels > 6:
        args.smoothness_levels = 6
    
    # ----- Load and enable the shape_generator addon -----
    print('-------LOADING SHAPE GENERATOR------')
    addon_file_path = os.path.join(os.getcwd(), 'shape_generator', 'operators.py')
    if os.path.exists(addon_file_path):
        try:
            bpy.ops.preferences.addon_install(filepath=addon_file_path)
            bpy.ops.preferences.addon_enable(module='shape_generator')
            bpy.ops.wm.save_userpref()
            print('-------FINISHED LOADING SHAPE GENERATOR------')
        except Exception as e:
            print(f"Error installing addon: {e}")
    else:
        print(f"Addon file not found at {addon_file_path}. Please ensure the addon is installed.")
    
    # Initialize nested stimulus_info dictionary
    stimulus_info = {}
    
    # ----- Handle preview mode -----
    if args.preview:
        print("\nRunning in PREVIEW mode")
        print("Generating 3 shapes for each combination of:")
        print("- n_extrusions: 2-9")
        print("- smoothness levels: 0-5")
        
        # Fixed settings for preview mode
        n_extrusions_range = range(2, 10)  # 2-9
        smoothness_levels = [0, 5]  # Full range 0-5
        shapes_per_config = 100  # 3 examples per configuration
        
        for n_ext in n_extrusions_range:
            print(f"\nGenerating preview for n_extrusions={n_ext}")
            extrusion_key = f"extrusions_{n_ext}"
            
            for level in smoothness_levels:
                
                subdivisions = level
                
                # Create level name
                level_name = f"smoothness_{level}"
                
                # Ensure nested structure exists
                ensure_nested_dict_structure(stimulus_info, extrusion_key, level_name)
                
                # Create output directory
                level_output = os.path.join(args.output, f"extrusions_{n_ext}", level_name)
                os.makedirs(level_output, exist_ok=True)
                
                print(f"  Generating {shapes_per_config} shapes with smoothness level {level} ({subdivisions} subdivisions)")
                
                for shape_idx in range(shapes_per_config):
                    # Generate random parameters
                    params = get_random_params(n_ext)
                    params['n_subdivisions'] = subdivisions
                    
                    # Process the shape
                    shape_name = process_shape(params, level_output, level_name, shape_idx, args)
                    
                    # Store in nested structure
                    stimulus_info[extrusion_key][level_name][shape_name] = params
    else:
        # ----- Normal mode -----
        print("\nRunning in NORMAL mode")
        
        # If n_extrusions wasn't specified, use a default value
        if args.n_extrusions is None:
            args.n_extrusions = 9
        
        print(f"Using parameters:")
        print(f"  n_extrusions:     {args.n_extrusions}")
        print(f"  output dir:       {args.output}")
        print(f"  n_shapes:         {args.n_shapes}")
        print(f"  smoothness levels: {args.smoothness_levels}")
        if args.textures:
            print(f"  texture dir:      {args.textures}")
        print("\n")
        
        extrusion_key = f"extrusions_{args.n_extrusions}"
        
        # Process each smoothness level
        for level in range(args.smoothness_levels):
            # Calculate subdivisions based on level
            if args.smoothness_levels == 1:
                subdivisions = 0  # Just blocky if only one level
            else:
                # Distribute levels evenly from 0 to 5
                subdivisions = int(5 * level / (args.smoothness_levels - 1))
            
            # Create appropriate level name
            if subdivisions == 0:
                level_name = "blocky"
            elif subdivisions == 5:
                level_name = "smooth"
            else:
                level_name = f"smoothness_{subdivisions}"
            
            # Ensure nested structure exists
            ensure_nested_dict_structure(stimulus_info, extrusion_key, level_name)
            
            # Create output directory
            level_output = os.path.join(args.output, f"extrusions_{args.n_extrusions}", level_name)
            os.makedirs(level_output, exist_ok=True)
            
            print(f"\nGenerating {args.n_shapes} shapes with {level_name} ({subdivisions} subdivisions)")
            
            shape_count = 0
            while shape_count < args.n_shapes:
                # Generate random parameters
                params = get_random_params(args.n_extrusions)
                params['n_subdivisions'] = subdivisions
                
                print(f"Generating {level_name} shape {shape_count+1}/{args.n_shapes}")
                
                # Process the shape
                shape_name = process_shape(params, level_output, level_name, shape_count, args)
                
                # Store in nested structure
                stimulus_info[extrusion_key][level_name][shape_name] = params
                
                shape_count += 1
    
    # ----- Save stimulus info -----
    with open(os.path.join(args.output, 'stimulus_info.pkl'), 'wb') as f:
        pickle.dump(stimulus_info, f)
    
    # Count total shapes for summary
    total_shapes = 0
    for extrusion_key in stimulus_info:
        for smoothness_key in stimulus_info[extrusion_key]:
            total_shapes += len(stimulus_info[extrusion_key][smoothness_key])
    
    print("\nGeneration complete!")
    print(f"Generated {total_shapes} shapes")
    print(f"Stimulus info saved to {os.path.join(args.output, 'stimulus_info.pkl')}")
    
    # Print structure summary
    print("\nStimulus info structure:")
    for extrusion_key in stimulus_info:
        print(f"  {extrusion_key}:")
        for smoothness_key in stimulus_info[extrusion_key]:
            num_shapes = len(stimulus_info[extrusion_key][smoothness_key])
            print(f"    {smoothness_key}: {num_shapes} shapes")