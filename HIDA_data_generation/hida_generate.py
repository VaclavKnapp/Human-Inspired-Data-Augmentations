#!/usr/bin/env python3
"""
HIDA Data Generation - Unified Pipeline
A streamlined interface for generating and rendering Shapegen and Primigen objects.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(step_num, total_steps, description):
    print(f"{Colors.CYAN}[Step {step_num}/{total_steps}]{Colors.END} {Colors.BOLD}{description}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def run_command(cmd, description, cwd=None, check=True):
    """Run a command and handle errors"""
    print_info(f"Running: {description}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {description}")
        if e.stderr:
            print(e.stderr)
        return False

def find_blender():
    """Find Blender executable"""
    common_paths = [
        "/home/vaclav_knapp/blender-3.6.19-linux-x64/blender",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "blender"
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    # Try to find in PATH
    result = subprocess.run(["which", "blender"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()

    return None

def setup_utilities(args):
    """Step 1: Generate viewpoints and textures if needed"""
    print_header("Setup Utilities")

    base_dir = Path(__file__).parent
    utils_dir = base_dir / "utils"

    # Check for viewpoints file
    if args.viewpoints_file and os.path.exists(args.viewpoints_file):
        print_success(f"Using existing viewpoints file: {args.viewpoints_file}")
    else:
        print_step(1, 2, "Generating Fibonacci sphere viewpoints")
        output_file = args.output_dir / "viewpoints" / f"{args.n_viewpoints}_points.pkl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3",
            str(utils_dir / "compute_sphere.py"),
            str(args.n_viewpoints),
            "--output", str(output_file)
        ]

        if run_command(cmd, "Generate viewpoints"):
            args.viewpoints_file = str(output_file)
            print_success(f"Created viewpoints file: {output_file}")
        else:
            print_error("Failed to generate viewpoints")
            return False

    # Check for textures
    if args.textures and os.path.exists(args.textures) and len(list(Path(args.textures).glob("*.png"))) > 0:
        print_success(f"Using existing textures from: {args.textures}")
    else:
        print_step(2, 2, "Generating procedural textures")
        texture_dir = args.output_dir / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)

        blender = find_blender()
        if not blender:
            print_error("Blender not found. Please install Blender or set BLENDER_PATH")
            return False

        cmd = [
            blender,
            "--background",
            "--python", str(utils_dir / "create_textures.py"),
            "--", str(args.n_textures)
        ]

        if run_command(cmd, "Generate textures", cwd=str(texture_dir)):
            args.textures = str(texture_dir)
            print_success(f"Created {args.n_textures * 8} textures in: {texture_dir}")
        else:
            print_error("Failed to generate textures")
            return False

    return True

def generate_shapegen(args):
    """Step 2: Generate Shapegen objects"""
    print_header("Generate Shapegen Objects")

    if not args.shapegen:
        print_info("Skipping Shapegen generation (--no-shapegen)")
        return True

    base_dir = Path(__file__).parent
    shapegen_dir = base_dir / "Shapegen"
    script = shapegen_dir / "Object_generation" / "generate_stim_final_w_pkl.py"

    output_dir = args.output_dir / "shapegen_objects"
    output_dir.mkdir(parents=True, exist_ok=True)

    blender = find_blender()
    if not blender:
        print_error("Blender not found")
        return False

    cmd = [
        blender,
        "--background",
        "--python", str(script),
        "--",
        "--output", str(output_dir),
        "--n_shapes", str(args.n_shapes),
        "--n_extrusions", str(args.extrusions),
        "--smoothness_levels", str(args.smoothness_levels)
    ]

    if args.textures:
        cmd.extend(["--textures", str(args.textures)])

    if args.preview:
        cmd.append("--preview")

    print_step(1, 1, f"Generating {args.n_shapes} Shapegen objects")
    if run_command(cmd, "Generate Shapegen objects"):
        print_success(f"Shapegen objects saved to: {output_dir}")
        args.shapegen_objects = output_dir
        return True
    else:
        print_error("Shapegen generation failed")
        return False

def generate_primigen(args):
    """Step 3: Generate Primigen objects"""
    print_header("Generate Primigen Objects")

    if not args.primigen:
        print_info("Skipping Primigen generation (--no-primigen)")
        return True

    base_dir = Path(__file__).parent
    primigen_dir = base_dir / "Primigen"
    script = primigen_dir / "Object_generation" / "primishapegen_v2.py"

    output_base = args.output_dir / "primigen_objects"
    output_base.mkdir(parents=True, exist_ok=True)

    blender = find_blender()
    if not blender:
        print_error("Blender not found")
        return False

    total_objects = 0

    # Generate hierarchy: n-level × config × warp × place
    for n_obj in args.n_primitives:
        print_step(n_obj - 1, len(args.n_primitives), f"Generating n={n_obj} primitive objects")

        n_dir = output_base / f"n{n_obj}"
        n_dir.mkdir(parents=True, exist_ok=True)

        for config_idx in range(1, args.n_configs + 1):
            # Generate random shape counts
            shape_counts = generate_shape_counts(n_obj)

            for warp_idx in range(1, args.n_warps + 1):
                for place_idx in range(1, args.n_placements + 1):
                    place_dir = n_dir / f"config_{config_idx}" / f"warp_{warp_idx}" / f"place_{place_idx}"
                    place_dir.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        blender,
                        "--background",
                        "--python", str(script),
                        "--",
                        "--num_shapes", "1",
                        "--output_dir", str(place_dir),
                        "--shape_counts", shape_counts,
                        "--seed", str((n_obj * 10000) + (config_idx * 1000) + (warp_idx * 100) + place_idx),
                        "--texture_folder", str(args.textures) if args.textures else ""
                    ]

                    if not run_command(cmd, f"n{n_obj}_c{config_idx}_w{warp_idx}_p{place_idx}", check=False):
                        print_error(f"Failed to generate object")
                        continue

                    total_objects += 1

    print_success(f"Generated {total_objects} Primigen objects in: {output_base}")
    args.primigen_objects = output_base
    return True

def generate_shape_counts(n_obj):
    """Generate random shape counts for a given number of objects"""
    import random

    # Generate random distribution
    cylinder = random.randint(0, n_obj)
    cube = random.randint(0, n_obj - cylinder)
    ellipsoid = n_obj - cylinder - cube

    return f"cylinder:{cylinder},cube:{cube},ellipsoid:{ellipsoid},shapegen:0"

def render_objects(args):
    """Step 4: Render generated objects"""
    print_header("Render Objects")

    base_dir = Path(__file__).parent

    objects_to_render = []
    if hasattr(args, 'shapegen_objects') and args.shapegen_objects:
        objects_to_render.append(('Shapegen', args.shapegen_objects, base_dir / "Shapegen" / "Rendering"))
    if hasattr(args, 'primigen_objects') and args.primigen_objects:
        objects_to_render.append(('Primigen', args.primigen_objects, base_dir / "Primigen" / "Rendering"))

    if not objects_to_render:
        print_info("No objects to render")
        return True

    for dataset_name, obj_dir, render_dir in objects_to_render:
        print_step(1, len(objects_to_render), f"Rendering {dataset_name} objects")

        output_dir = args.output_dir / f"{dataset_name.lower()}_renders"
        output_dir.mkdir(parents=True, exist_ok=True)

        render_script = render_dir / "render_simplified.sh"

        cmd = [
            "bash",
            str(render_script),
            str(obj_dir),
            str(output_dir),
            f"--{args.n_lights}_lights",
            "--random_dim",
            "--sphere=true",
            f"--sphere_file={args.viewpoints_file}",
            "--scale_camera=true"
        ]

        if run_command(cmd, f"Render {dataset_name}"):
            print_success(f"{dataset_name} renders saved to: {output_dir}")
        else:
            print_error(f"{dataset_name} rendering failed")
            return False

    return True

def save_config(args):
    """Save pipeline configuration for reproducibility"""
    config_file = args.output_dir / "pipeline_config.json"

    config = {
        "pipeline_version": "1.0",
        "datasets": {
            "shapegen": args.shapegen,
            "primigen": args.primigen
        },
        "shapegen_params": {
            "n_shapes": args.n_shapes,
            "extrusions": args.extrusions,
            "smoothness_levels": args.smoothness_levels,
            "preview": args.preview
        },
        "primigen_params": {
            "n_primitives": args.n_primitives,
            "n_configs": args.n_configs,
            "n_warps": args.n_warps,
            "n_placements": args.n_placements
        },
        "rendering": {
            "n_viewpoints": args.n_viewpoints,
            "n_lights": args.n_lights
        },
        "paths": {
            "output_dir": str(args.output_dir),
            "textures": str(args.textures) if args.textures else None,
            "viewpoints": str(args.viewpoints_file) if args.viewpoints_file else None
        }
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print_success(f"Pipeline configuration saved to: {config_file}")

def main():
    parser = argparse.ArgumentParser(
        description="HIDA Data Generation - Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start - generate everything with defaults
  python hida_generate.py --output my_dataset

  # Generate only Shapegen with preview mode
  python hida_generate.py --output my_dataset --no-primigen --preview

  # Custom Primigen hierarchy
  python hida_generate.py --output my_dataset --no-shapegen \\
      --n-primitives 2 3 4 --n-configs 3 --n-warps 3 --n-placements 5

  # Use existing textures and viewpoints
  python hida_generate.py --output my_dataset \\
      --textures /path/to/textures \\
      --viewpoints-file /path/to/viewpoints.pkl
        """
    )

    # Required arguments
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for all generated data')

    # Dataset selection
    parser.add_argument('--no-shapegen', dest='shapegen', action='store_false',
                        help='Skip Shapegen generation')
    parser.add_argument('--no-primigen', dest='primigen', action='store_false',
                        help='Skip Primigen generation')

    # Shapegen parameters
    shapegen_group = parser.add_argument_group('Shapegen parameters')
    shapegen_group.add_argument('--n-shapes', type=int, default=100,
                                help='Number of shapes per smoothness level (default: 100)')
    shapegen_group.add_argument('--extrusions', type=int, default=9,
                                help='Number of extrusions (default: 9)')
    shapegen_group.add_argument('--smoothness-levels', type=int, default=2,
                                help='Number of smoothness levels (default: 2)')
    shapegen_group.add_argument('--preview', action='store_true',
                                help='Generate preview set (all extrusions × smoothness)')

    # Primigen parameters
    primigen_group = parser.add_argument_group('Primigen parameters')
    primigen_group.add_argument('--n-primitives', type=int, nargs='+', default=[2, 3, 4],
                                help='Number of primitives per object (default: 2 3 4)')
    primigen_group.add_argument('--n-configs', type=int, default=6,
                                help='Number of configurations per n-level (default: 6)')
    primigen_group.add_argument('--n-warps', type=int, default=6,
                                help='Number of warps per config (default: 6)')
    primigen_group.add_argument('--n-placements', type=int, default=15,
                                help='Number of placements per warp (default: 15)')

    # Rendering parameters
    render_group = parser.add_argument_group('Rendering parameters')
    render_group.add_argument('--n-viewpoints', type=int, default=50,
                             help='Number of viewpoints per object (default: 50)')
    render_group.add_argument('--n-lights', type=int, default=8, choices=[4, 6, 8, 16],
                             help='Number of lights (default: 8)')

    # Utilities
    utils_group = parser.add_argument_group('Utility parameters')
    utils_group.add_argument('--textures', type=str,
                            help='Path to existing texture directory')
    utils_group.add_argument('--n-textures', type=int, default=100,
                            help='Number of textures to generate per category (default: 100)')
    utils_group.add_argument('--viewpoints-file', type=str,
                            help='Path to existing viewpoints .pkl file')

    # Pipeline control
    parser.add_argument('--skip-setup', action='store_true',
                        help='Skip utility setup (viewpoints/textures)')
    parser.add_argument('--skip-render', action='store_true',
                        help='Only generate objects, skip rendering')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("HIDA Data Generation Pipeline")
    print_info(f"Output directory: {args.output_dir}")
    print_info(f"Datasets: Shapegen={args.shapegen}, Primigen={args.primigen}")

    # Pipeline steps
    steps = []
    if not args.skip_setup:
        steps.append(("Setup utilities", setup_utilities))
    if args.shapegen:
        steps.append(("Generate Shapegen", generate_shapegen))
    if args.primigen:
        steps.append(("Generate Primigen", generate_primigen))
    if not args.skip_render:
        steps.append(("Render objects", render_objects))

    # Execute pipeline
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n{Colors.BOLD}Step {i}/{len(steps)}: {step_name}{Colors.END}")
        if not step_func(args):
            print_error(f"Pipeline failed at step: {step_name}")
            return 1

    # Save configuration
    save_config(args)

    print_header("Pipeline Complete!")
    print_success(f"All outputs saved to: {args.output_dir}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
