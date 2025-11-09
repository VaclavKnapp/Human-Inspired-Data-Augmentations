#!/bin/bash
#Code 1 - Simplified version that saves camera info


light_setup="8_lights"
random_dim=true
preview=false
light_colors="white"  # Default light color (white)
video=false          # Default is to generate images, not video
sphere=true        # Fibonacci sphere rendering mode
sphere_file="/home/vaclav_knapp/3D_datagen_preview_2/50_points.pkl"      # Path to the Fibonacci sphere viewpoints file
scale_camera=true   # Default is to not scale camera based on object size


if [ "$#" -lt 2 ]; then
    echo "Usage: ./render_transparent.sh /path/to/blend/folder /path/to/output/folder [options]"
    echo "Options include: --4_lights/--6_lights/--8_lights/--16_lights, --random_dim, --preview,"
    echo "                 --light_colors=COLOR, --video=true/false, --sphere=true --sphere_file=/path/to/viewpoints.pkl"
    echo "                 --scale_camera=true/false"
    exit 1
fi


blend_dir="$1"
output_dir="$2"


shift 2


while [ "$#" -gt 0 ]; do
    case "$1" in
        --4_lights)
            light_setup="4_lights"
            ;;
        --6_lights)
            light_setup="6_lights"
            ;;
        --8_lights)
            light_setup="8_lights"
            ;;
        --16_lights)
            light_setup="16_lights"
            ;;
        --random_dim)
            random_dim=true
            ;;
        --preview)
            preview=true
            ;;
        --light_colors=*)
            light_colors="${1#*=}"
            ;;
        --video=*)
            video="${1#*=}"
            ;;
        --sphere=*)
            sphere_value="${1#*=}"
            if [ "$sphere_value" = "true" ]; then
                sphere=true
            fi
            ;;
        --sphere_file=*)
            sphere_file="${1#*=}"
            ;;
        --scale_camera=*)
            scale_camera="${1#*=}"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


if [ ! -d "$blend_dir" ]; then
    echo "Error: Blend directory '$blend_dir' does not exist."
    exit 1
fi

if [ "$sphere" = true ] && [ ! -f "$sphere_file" ]; then
    echo "Error: Sphere viewpoints file '$sphere_file' does not exist."
    exit 1
fi


mkdir -p "$output_dir"
# Create camera info directory
camera_info_dir="${output_dir}_camera_info"
mkdir -p "$camera_info_dir"




python_script=$(mktemp)


trap 'rm -f "$python_script"' EXIT


cat <<'EOL' > "$python_script"
import bpy
import random
import math
import os
import sys
import mathutils
import pickle


argv = sys.argv
argv = argv[argv.index("--") + 1:]  


object_name = None
output_dir = None
camera_info_dir = None
light_setup = "16_lights"  
random_dim = False
preview = False
light_colors = "white"  # default light color
video = False          # default is to generate images, not video
sphere = False         # Fibonacci sphere rendering mode
sphere_file = None     # Path to the Fibonacci sphere viewpoints file
scale_camera = False   # Scale camera based on object size


i = 0
while i < len(argv):
    arg = argv[i]
    if arg == '--random_dim':
        random_dim = True
        i += 1
    elif arg == '--preview':
        preview = True
        i += 1
    elif arg in ('--4_lights', '--6_lights', '--8_lights', '--16_lights'):
        light_setup = arg.lstrip('--')
        i += 1
    elif arg.startswith('--light_colors='):
        light_colors = arg.split('=', 1)[1]
        i += 1
    elif arg.startswith('--video='):
        video_value = arg.split('=', 1)[1].lower()
        video = (video_value == 'true')
        i += 1
    elif arg.startswith('--sphere='):
        sphere_value = arg.split('=', 1)[1].lower()
        sphere = (sphere_value == 'true')
        i += 1
    elif arg.startswith('--sphere_file='):
        sphere_file = arg.split('=', 1)[1]
        i += 1
    elif arg.startswith('--scale_camera='):
        scale_camera_value = arg.split('=', 1)[1].lower()
        scale_camera = (scale_camera_value == 'true')
        i += 1
    elif arg.startswith('--camera_info_dir='):
        camera_info_dir = arg.split('=', 1)[1]
        i += 1
    else:
        if object_name is None:
            object_name = arg
        elif output_dir is None:
            output_dir = arg
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
        i +=1


if object_name is None or output_dir is None:
    print("Usage: blender -b <file.blend> --python <script.py> -- <object_name> <output_dir> [options]")
    sys.exit(1)

# Set camera_info_dir if not provided
if camera_info_dir is None:
    camera_info_dir = output_dir + "_camera_info"


def get_scene_bounds():
    """
    Calculate the bounding box of all visible mesh objects in the scene.
    Returns the maximum dimension and the center point of the bounding box.
    """
    min_coord = mathutils.Vector((float('inf'), float('inf'), float('inf')))
    max_coord = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
    
    has_objects = False
    
    # Iterate through all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.visible_get():
            has_objects = True
            # Get world-space bounding box
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            for corner in bbox_corners:
                min_coord.x = min(min_coord.x, corner.x)
                min_coord.y = min(min_coord.y, corner.y)
                min_coord.z = min(min_coord.z, corner.z)
                max_coord.x = max(max_coord.x, corner.x)
                max_coord.y = max(max_coord.y, corner.y)
                max_coord.z = max(max_coord.z, corner.z)
    
    if not has_objects:
        print("Warning: No visible mesh objects found in scene")
        return 10.0, mathutils.Vector((0, 0, 0))  # Default values
    
    # Calculate dimensions
    dimensions = max_coord - min_coord
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    
    # Calculate center
    center = (min_coord + max_coord) / 2
    
    print(f"Scene bounds: dimensions={dimensions}, max_dimension={max_dimension}, center={center}")
    
    return max_dimension, center


def calculate_camera_radius(base_radius, scale_camera):
    """
    Calculate the appropriate camera radius based on object size if scale_camera is True.
    """
    if not scale_camera:
        return base_radius, mathutils.Vector((0, 0, 0))
    
    max_dimension, center = get_scene_bounds()
    
    # Calculate a scaling factor
    # We want the camera to be at a distance proportional to the object size
    # Using a factor of 2.5-3.0 usually gives good framing
    scale_factor = 2.5
    scaled_radius = max_dimension * scale_factor
    
    # Apply minimum and maximum limits
    min_radius = 5.0
    max_radius = 100.0
    scaled_radius = max(min_radius, min(max_radius, scaled_radius))
    
    print(f"Camera scaling: base_radius={base_radius}, scaled_radius={scaled_radius}, object_center={center}")
    
    return scaled_radius, center


def spherical_to_cartesian(radius, theta, phi):
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)
    return (x, y, z)


def look_at(obj_camera, point):
    direction = point - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


def generate_sphere_points(n_points, radius):
    points = []
    offset = 2.0 / n_points
    increment = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    for i in range(n_points):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y * y)
        phi = i * increment
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        points.append((x * radius, y * radius, z * radius))
    return points


def load_viewpoints(file_path, radius=10.0):
    """
    Load viewpoints from a pickle file and scale them to the desired radius.
    
    Args:
        file_path (str): Path to the pickle file containing viewpoints
        radius (float): Radius to scale the viewpoints to
        
    Returns:
        list: List of (x, y, z) coordinates scaled to the specified radius
    """
    try:
        with open(file_path, 'rb') as f:
            viewpoints = pickle.load(f)
        
        # Scale the unit viewpoints to the desired radius
        scaled_viewpoints = [(x * radius, y * radius, z * radius) for x, y, z in viewpoints]
        return scaled_viewpoints
    except Exception as e:
        print(f"Error loading viewpoints file: {e}")
        return None


# Get color from name or generate random color
def get_light_color(color_name):
    # Define predefined colors
    colors = {
        "white": (1.0, 1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "orange": (1.0, 0.5, 0.0, 1.0),
        "purple": (0.5, 0.0, 0.5, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "pink": (1.0, 0.75, 0.8, 1.0),
        "teal": (0.0, 0.5, 0.5, 1.0)
    }
    
    if color_name.lower() == "random":
        # Generate a random bright color
        r = random.uniform(0.5, 1.0)
        g = random.uniform(0.5, 1.0)
        b = random.uniform(0.5, 1.0)
        return (r, g, b, 1.0)
    elif color_name.lower() in colors:
        return colors[color_name.lower()]
    else:
        # Default to white if color not recognized
        print(f"Unrecognized color: {color_name}, using white")
        return colors["white"]


def setup_transparent_background():
    """
    Set up the scene for transparent background rendering.
    """
    # Make sure we're using a transparent background for the 3D scene
    bpy.context.scene.render.film_transparent = True
    
    # Set up the world background to be transparent
    world = bpy.context.scene.world
    world.use_nodes = True
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)
    
    bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = (0, 0, 0, 0)  # Transparent background
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    # Set up compositing for direct output with transparency
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    
    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create input node (renders the 3D scene with transparency)
    render_node = tree.nodes.new(type='CompositorNodeRLayers')
    
    # Output node
    output_node = tree.nodes.new(type='CompositorNodeComposite')
    
    # Connect nodes directly - no background
    tree.links.new(render_node.outputs['Image'], output_node.inputs['Image'])
    
    # Position nodes for clarity
    render_node.location = (0, 0)
    output_node.location = (300, 0)


def remove_lights(lights):
    """Safely remove lights, handling cases where they might have been removed already."""
    for light in lights[:]:  # Make a copy of the list to safely iterate
        try:
            if light.name in bpy.data.objects:
                bpy.data.objects.remove(light, do_unlink=True)
        except ReferenceError:
            # Skip if the object has already been removed
            pass


def create_lighting_setup(light_setup, base_radius, random_dim=False, randomize_positions=False, object_center=None):
    """
    Create lighting setup with options to randomize positions and intensities.
    
    Args:
        light_setup: String indicating number of lights (e.g., "4_lights")
        base_radius: Base radius for positioning lights
        random_dim: Whether to randomize light intensities
        randomize_positions: Whether to randomize light positions (uses different seed each time)
        object_center: Center point of the object (for offsetting lights)
    
    Returns:
        List of created light objects
    """
    # Remove any existing lights first
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()
    
    if object_center is None:
        object_center = mathutils.Vector((0, 0, 0))
    
    lights = []
    used_intensities = []  # To track already used intensities
    
    if light_setup in ('4_lights', '6_lights', '8_lights', '16_lights'):
        n_lights = int(light_setup.split('_')[0])  # e.g. '6_lights' -> 6
        light_radius = base_radius
        
        # Randomize positions if requested (different seed each time)
        if randomize_positions:
            random_seed = random.randint(0, 1000000)
            random.seed(random_seed)
        
        light_positions = generate_sphere_points(n_lights, light_radius)
        
        # Energy range - scale based on object size if camera is scaled
        max_energy = 1350
        if scale_camera:
            # Scale light energy based on the camera radius
            energy_scale = base_radius / 10.0  # Assuming 10 is the default base radius
            max_energy = max_energy * energy_scale
        
        min_energy = max_energy * 0.20
        min_diff = 200  # Minimum difference between light intensities
        
        for pos in light_positions:
            # Offset light position by object center
            adjusted_pos = (pos[0] + object_center.x, pos[1] + object_center.y, pos[2] + object_center.z)
            bpy.ops.object.light_add(type='POINT', location=adjusted_pos)
            light = bpy.context.object
            
            # Set light color based on light_colors option
            if light_colors.lower() == "random":
                # Each light gets a different random color
                light.data.color = get_light_color("random")[:3]  # Exclude alpha
            else:
                # All lights get the same specified color
                light.data.color = get_light_color(light_colors)[:3]  # Exclude alpha
            
            if random_dim:
                # Try to find an intensity that's different enough from existing ones
                intensity = 0
                best_intensity = random.uniform(min_energy, max_energy)
                best_diff = 0
                
                # Try up to 20 times to find a good intensity
                for _ in range(20):
                    candidate = random.uniform(min_energy, max_energy)
                    # If no intensities yet, just use this one
                    if not used_intensities:
                        intensity = candidate
                        break
                        
                    # Check if this candidate is different enough from all existing intensities
                    is_valid = True
                    min_found_diff = float('inf')
                    
                    for used in used_intensities:
                        diff = abs(candidate - used)
                        min_found_diff = min(min_found_diff, diff)
                        if diff < min_diff:
                            is_valid = False
                            break
                    
                    # If valid, use it
                    if is_valid:
                        intensity = candidate
                        break
                    # Otherwise track the best one we found
                    elif min_found_diff > best_diff:
                        best_diff = min_found_diff
                        best_intensity = candidate
                
                # If we couldn't find a good one after all attempts, use the best we found
                if intensity == 0:
                    intensity = best_intensity
                
                # Store the chosen intensity
                used_intensities.append(intensity)
                light.data.energy = intensity
            else:
                light.data.energy = max_energy
                
            lights.append(light)
    else:
        # Basic three-point lighting
        light_radius = base_radius

        # Define energy ranges for each light
        energy_ranges = [
            (7500, 10000),  # Key light
            (3500, 5000),   # Fill light
            (6000, 8000)    # Back light
        ]
        
        # Scale energy ranges if camera is scaled
        if scale_camera:
            energy_scale = base_radius / 10.0
            energy_ranges = [(min_e * energy_scale, max_e * energy_scale) for min_e, max_e in energy_ranges]

        # Generate positions (could be randomized in future)
        positions = [
            spherical_to_cartesian(light_radius, math.radians(60), math.radians(30)),   # Key light
            spherical_to_cartesian(light_radius, math.radians(70), math.radians(150)),  # Fill light
            spherical_to_cartesian(light_radius, math.radians(110), math.radians(-90))  # Back light
        ]
        
        # Create lights with specific intensity ranges
        for i, (pos, energy_range) in enumerate(zip(positions, energy_ranges)):
            # Offset light position by object center
            adjusted_pos = (pos[0] + object_center.x, pos[1] + object_center.y, pos[2] + object_center.z)
            bpy.ops.object.light_add(type='POINT', location=adjusted_pos)
            light = bpy.context.object
            
            # Set light color based on light_colors option
            if light_colors.lower() == "random":
                # Each light gets a different random color
                light.data.color = get_light_color("random")[:3]  # Exclude alpha
            else:
                # All lights get the same specified color
                light.data.color = get_light_color(light_colors)[:3]  # Exclude alpha
            
            if random_dim:
                min_e, max_e = energy_range
                
                # Try to find an intensity different from existing ones
                intensity = 0
                best_intensity = random.uniform(min_e, max_e)
                
                # Try up to 20 times to find a good intensity
                for _ in range(20):
                    candidate = random.uniform(min_e, max_e)
                    
                    # If no intensities yet, just use this one
                    if not used_intensities:
                        intensity = candidate
                        break
                        
                    # Check if different enough from existing intensities
                    is_valid = True
                    for used in used_intensities:
                        if abs(candidate - used) < min_diff:
                            is_valid = False
                            break
                    
                    # If valid, use it
                    if is_valid:
                        intensity = candidate
                        break
                
                # If we couldn't find a good one, use the best we found
                if intensity == 0:
                    intensity = best_intensity
                
                # Store the chosen intensity
                used_intensities.append(intensity)
                light.data.energy = intensity
            else:
                # Use default values if not randomizing
                light.data.energy = energy_ranges[i][1]  # Use max value
                
            lights.append(light)
    
    return lights


def generate_camera_position(radius, object_center=None):
    """
    Generate a random camera position on the sphere - simplified version without complex constraints.
    """
    if object_center is None:
        object_center = mathutils.Vector((0, 0, 0))
        
    # Generate random spherical coordinates - full sphere
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    
    # Convert to cartesian coordinates
    x, y, z = spherical_to_cartesian(radius, theta, phi)
    
    # Add object center offset
    camera_pos = (x + object_center.x, y + object_center.y, z + object_center.z)
    
    return camera_pos, theta, phi


def save_camera_info(camera_position, camera_rotation, object_center, camera_radius, theta, phi, output_path):
    """
    Save camera information to a pickle file.
    """
    # Get camera object
    camera = bpy.context.scene.camera
    
    # Get camera intrinsics
    scene = bpy.context.scene
    render = scene.render
    camera_data = camera.data
    
    # Calculate focal length in pixels
    sensor_width = camera_data.sensor_width
    focal_length_mm = camera_data.lens
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    
    # Convert focal length to pixels
    focal_length_px = (focal_length_mm * resolution_x) / sensor_width
    
    camera_info = {
        'position': list(camera_position),
        'rotation_euler': list(camera_rotation),
        'object_center': list(object_center),
        'camera_radius': camera_radius,
        'theta': theta,
        'phi': phi,
        'intrinsics': {
            'focal_length_mm': focal_length_mm,
            'focal_length_px': focal_length_px,
            'sensor_width': sensor_width,
            'resolution_x': resolution_x,
            'resolution_y': resolution_y,
            'principal_point': [resolution_x / 2, resolution_y / 2]  # Assuming centered
        },
        'timestamp': str(bpy.app.build_date)
    }
    
    # Save to pickle file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(camera_info, f)


# Video creation function
def setup_video_rendering(object_name, output_dir, camera_radius, object_center=None):
    """
    Set up rendering for a video with the camera circling around the object.
    """
    if object_center is None:
        object_center = mathutils.Vector((0, 0, 0))
        
    # Create output directory
    video_output_dir = os.path.join(output_dir)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Set output file path for the video
    output_file = os.path.join(video_output_dir, f"{object_name}_video.mp4")
    
    # Set up transparent background
    setup_transparent_background()
    
    # Set render settings for video
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.filepath = output_file
    
    # Set animation length (120 frames = 5 seconds at 24 fps)
    scene.frame_start = 1
    scene.frame_end = 80
    scene.render.fps = 20
    
    # Create a circle path for the camera
    # Choose a vertical offset for more interesting angles
    vertical_offset = camera_radius * 0.3 + object_center.z
    
    # Clear any existing animation
    if bpy.context.scene.camera.animation_data:
        bpy.context.scene.camera.animation_data_clear()
    
    # Create the camera path keyframes
    for frame in range(1, 81):
        # Calculate angle (in radians) for this frame
        angle = frame * (2 * math.pi / 80)  # Full 360-degree rotation
        
        # Calculate camera position
        x = camera_radius * math.cos(angle) + object_center.x
        y = camera_radius * math.sin(angle) + object_center.y
        z = vertical_offset
        
        # Set camera position
        bpy.context.scene.camera.location = (x, y, z)
        
        # Make camera look at the object center
        look_at(bpy.context.scene.camera, object_center)
        
        # Insert keyframe for location and rotation
        bpy.context.scene.camera.keyframe_insert(data_path="location", frame=frame)
        bpy.context.scene.camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    return output_file


# Set up the renderer
bpy.context.scene.render.resolution_x = 518
bpy.context.scene.render.resolution_y = 518


# Ensure we have a camera
if bpy.context.scene.camera is None:
    bpy.ops.object.camera_add()
    bpy.context.scene.camera = bpy.context.object


# Delete existing lights
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()


# Calculate base radius and object center based on scale_camera setting
base_radius = 10
if scale_camera:
    base_radius, object_center = calculate_camera_radius(base_radius, scale_camera)
else:
    object_center = mathutils.Vector((0, 0, 0))


# Determine rendering mode
if sphere and sphere_file:
    # Fibonacci sphere rendering mode
    # Load the viewpoints
    camera_radius = base_radius * 1.1
    viewpoints = load_viewpoints(sphere_file, camera_radius)
    
    if not viewpoints:
        print("Error: Failed to load viewpoints from file.")
        sys.exit(1)
    
    # Render from each viewpoint
    for i, camera_position in enumerate(viewpoints):
        # Setup transparent background
        setup_transparent_background()
        
        # Adjust camera position by object center
        adjusted_camera_pos = (
            camera_position[0] + object_center.x,
            camera_position[1] + object_center.y,
            camera_position[2] + object_center.z
        )
        
        # Set camera position
        bpy.context.scene.camera.location = adjusted_camera_pos
        look_at(bpy.context.scene.camera, object_center)
        
        # Save camera info
        camera_info_path = os.path.join(camera_info_dir, f"{i:03d}.pkl")
        save_camera_info(
            adjusted_camera_pos, 
            bpy.context.scene.camera.rotation_euler, 
            object_center, 
            camera_radius, 
            0, 0,  # theta, phi not applicable for predetermined viewpoints
            camera_info_path
        )
        
        # Create new lights for each viewpoint
        lights = create_lighting_setup(light_setup, base_radius, random_dim, randomize_positions=False, object_center=object_center)
        
        # Set output path and render
        output_file = os.path.join(output_dir, f"{i:03d}.png")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        bpy.context.scene.render.filepath = output_file
        bpy.ops.render.render(write_still=True)
        
        # Remove lights
        remove_lights(lights)
        
    print(f"Fibonacci sphere rendering complete! {len(viewpoints)} images saved to {output_dir}")

elif video:
    # Video generation mode
    camera_radius = base_radius * 0.90
    
    # Setup video rendering
    output_file = setup_video_rendering(object_name, output_dir, camera_radius, object_center)
    
    # Create lighting setup - using white lights by default, or the specified color
    lights = create_lighting_setup(light_setup, base_radius, random_dim=True, object_center=object_center)
    
    # Render animation
    bpy.ops.render.render(animation=True)
    
    # Clean up
    remove_lights(lights)
    
    print(f"Video rendering complete! Video saved to {output_file}")

elif not preview:
    # Original rendering logic - 10 images (simplified, no complex constraints)
    for i in range(10):
        # Setup transparent background
        setup_transparent_background()

        # Set up camera
        camera_radius = base_radius * 0.90
        
        camera_position, theta, phi = generate_camera_position(camera_radius, object_center)

        bpy.context.scene.camera.location = camera_position
        look_at(bpy.context.scene.camera, object_center)
        
        # Save camera info
        camera_info_path = os.path.join(camera_info_dir, f"{i:03d}.pkl")
        save_camera_info(
            camera_position, 
            bpy.context.scene.camera.rotation_euler, 
            object_center, 
            camera_radius, 
            theta, phi,
            camera_info_path
        )

        # Add lighting
        lights = create_lighting_setup(light_setup, base_radius, random_dim, randomize_positions=False, object_center=object_center)

        # Set output path and render
        output_file = os.path.join(output_dir, f"{i:03d}.png")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        bpy.context.scene.render.filepath = output_file
        bpy.ops.render.render(write_still=True)

        # Remove lights
        remove_lights(lights)

else:
    # Preview mode with 3 variations of 5 images each (simplified)
    camera_radius = base_radius * 0.90
    
    # Define the preview modes
    preview_modes = [
        "camera_light",  # Both camera and lights changing
        "light",         # Fixed camera, changing lights
        "camera"         # Changing camera, fixed lights
    ]
    
    # Generate a fixed camera position for modes that use it
    fixed_cam_pos, fixed_theta, fixed_phi = generate_camera_position(camera_radius, object_center)
    
    # Create fixed lighting setup for modes that use it
    fixed_lights = create_lighting_setup(light_setup, base_radius, random_dim=True, randomize_positions=False, object_center=object_center)
    
    # For each preview mode
    for mode in preview_modes:
        # We'll create a subdirectory for each mode
        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        
        mode_camera_info_dir = os.path.join(camera_info_dir, mode)
        os.makedirs(mode_camera_info_dir, exist_ok=True)
        
        # Create clean lighting state between modes
        if mode != "camera_light" and mode != "light":
            # If we need fixed lights for this mode, recreate them
            # This ensures they haven't been removed by previous modes
            remove_lights(fixed_lights)
            fixed_lights = create_lighting_setup(light_setup, base_radius, random_dim=True, randomize_positions=False, object_center=object_center)
        
        for i in range(5):  # 5 images per mode
            # Setup transparent background
            setup_transparent_background()
            
            if mode == "camera_light":
                # Both camera and lights change
                cam_pos, theta, phi = generate_camera_position(camera_radius, object_center)
                    
                # Set camera
                bpy.context.scene.camera.location = cam_pos
                look_at(bpy.context.scene.camera, object_center)
                
                # Create new lights with random positions and intensities
                lights = create_lighting_setup(light_setup, base_radius, random_dim=True, randomize_positions=True, object_center=object_center)
                
            elif mode == "light":
                # Fixed camera, changing lights
                # Use fixed camera
                bpy.context.scene.camera.location = fixed_cam_pos
                look_at(bpy.context.scene.camera, object_center)
                
                cam_pos = fixed_cam_pos
                theta, phi = fixed_theta, fixed_phi
                
                # Create new lights with random positions and intensities
                lights = create_lighting_setup(light_setup, base_radius, random_dim=True, randomize_positions=True, object_center=object_center)
                
            elif mode == "camera":
                # Changing camera, fixed lights
                cam_pos, theta, phi = generate_camera_position(camera_radius, object_center)
                    
                # Set camera
                bpy.context.scene.camera.location = cam_pos
                look_at(bpy.context.scene.camera, object_center)
                
                # Use the fixed lights (which were recreated at the start of this mode)
                lights = fixed_lights
            
            # Save camera info
            camera_info_path = os.path.join(mode_camera_info_dir, f"{i:03d}.pkl")
            save_camera_info(
                cam_pos, 
                bpy.context.scene.camera.rotation_euler, 
                object_center, 
                camera_radius, 
                theta, phi,
                camera_info_path
            )
            
            # Set output path
            output_file = os.path.join(mode_dir, f"{i:03d}.png")
            
            # Render
            bpy.context.scene.render.filepath = output_file
            bpy.ops.render.render(write_still=True)
            
            # If we're using temporary lights, clean them up
            if mode in ["camera_light", "light"]:
                remove_lights(lights)
    
    # Final cleanup
    if fixed_lights:
        remove_lights(fixed_lights)

EOL


for n_dir in "$blend_dir"/*; do  # n2, n3, n4 directories
    if [ -d "$n_dir" ]; then
        n_basename=$(basename "$n_dir")
        echo "Processing object number directory: $n_basename"
        
        for config_dir in "$n_dir"/*; do  # config_X directories
            if [ -d "$config_dir" ]; then
                config_basename=$(basename "$config_dir")
                echo "  Processing config: $config_basename"
                
                for warp_dir in "$config_dir"/*; do  # warp_X directories
                    if [ -d "$warp_dir" ]; then
                        warp_basename=$(basename "$warp_dir")
                        echo "    Processing warp: $warp_basename"
                        
                        for place_dir in "$warp_dir"/*; do  # place_X directories
                            if [ -d "$place_dir" ]; then
                                place_basename=$(basename "$place_dir")
                                echo "      Processing place: $place_basename"
                                
                                # Find blend files in this directory
                                shopt -s nullglob
                                blend_files=( "$place_dir"/*.blend )
                                shopt -u nullglob
                                
                                if [ ${#blend_files[@]} -eq 0 ]; then
                                    echo "        No .blend files found in $place_dir"
                                    continue
                                fi
                                
                                # Create corresponding output directory structure
                                relative_path=$(realpath --relative-to="$blend_dir" "$place_dir")
                                target_dir="$output_dir/$relative_path"
                                mkdir -p "$target_dir"
                                
                                # Create camera info directory structure
                                camera_info_target_dir="$camera_info_dir/$relative_path"
                                mkdir -p "$camera_info_target_dir"
                                
                                for blend_file in "${blend_files[@]}"; do
                                    object_name=$(basename "$blend_file" .blend)
                                    echo "        Rendering $object_name..."
                                    
                                    blender_args=(
                                        "/home/vaclav_knapp/blender-3.6.19-linux-x64/blender"
                                        "-b" 
                                        "$blend_file" 
                                        "--python" 
                                        "$python_script" 
                                        "--" 
                                        "$object_name" 
                                        "$target_dir"
                                    )
                                    
                                    blender_args+=("--camera_info_dir=$camera_info_target_dir")

                                    if [ -n "$light_setup" ]; then
                                        blender_args+=("--$light_setup")
                                    fi

                                    if [ "$random_dim" = true ]; then
                                        blender_args+=("--random_dim")
                                    fi
                                    
                                    if [ "$preview" = true ]; then
                                        blender_args+=("--preview")
                                    fi
                                    
                                    if [ "$light_colors" != "white" ]; then
                                        blender_args+=("--light_colors=$light_colors")
                                    fi
                                    
                                    if [ "$video" = true ]; then
                                        blender_args+=("--video=true")
                                    fi
                                    
                                    if [ "$sphere" = true ]; then
                                        blender_args+=("--sphere=true" "--sphere_file=$sphere_file")
                                    fi
                                    
                                    if [ "$scale_camera" = true ]; then
                                        blender_args+=("--scale_camera=true")
                                    fi
                                    
                                    "${blender_args[@]}"
                                done
                            fi
                        done
                    fi
                done
            fi
        done
    fi
done

echo "Transparent rendering complete! All files saved to $output_dir"
echo "Camera info saved to $camera_info_dir"