#!/usr/bin/env python3
"""
Precompute valid image pairs under camera constraints and record the unique images
actually used **after** optional per-object subsampling.

Fix: the global unique-image list is computed only from the kept pairs (after sampling),
so changing --max_pairs_per_object from 50 to 20 will change the unique count when
those extra pairs introduced new images.
"""

import os
import pickle
import math
import json
import argparse
import random
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

# ---------------------------- I/O helpers ---------------------------- #

def load_camera_info(pkl_path: str) -> Dict:
    """Load camera information from a pickle file. Returns None on failure."""
    try:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


def get_camera_info_path(image_path: str, camera_base_path: str, image_base_path: str) -> str:
    """Convert an image path to its corresponding camera-info .pkl path."""
    rel_path = os.path.relpath(image_path, image_base_path)
    rel_dir = os.path.dirname(rel_path)
    filename = os.path.basename(rel_path)
    camera_filename = filename.replace('.png', '.pkl')
    return os.path.join(camera_base_path, rel_dir, camera_filename)


def extract_shape_id(shape_folder_name: str) -> str:
    return shape_folder_name.split('_')[-1]


def get_image_structure(base_path: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Traverse the directory structure:
    base_path/
      extrusions_X/
        smoothness_Y/
          shape_ZZZ/
            *.png
    Returns a nested dict: {extrusion: {smoothness: {shape_id: [image_paths]}}}
    """
    print("Parsing image directory structure...")
    image_structure: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    extrusion_dirs = [d for d in os.listdir(base_path) if d.startswith('extrusions_')]

    for extrusion_dir in tqdm(extrusion_dirs, desc="Processing extrusion levels"):
        extrusion_level = extrusion_dir.split('_')[1]
        extrusion_path = os.path.join(base_path, extrusion_dir)
        if not os.path.isdir(extrusion_path):
            continue

        image_structure[extrusion_level] = {}
        smoothness_dirs = [d for d in os.listdir(extrusion_path) if d.startswith('smoothness_')]

        for smoothness_dir in smoothness_dirs:
            smoothness_level = smoothness_dir.split('_')[1]
            smoothness_path = os.path.join(extrusion_path, smoothness_dir)
            if not os.path.isdir(smoothness_path):
                continue

            shape_groups: Dict[str, List[str]] = {}
            shape_dirs = [d for d in os.listdir(smoothness_path) if d.startswith('shape_')]

            for shape_dir in shape_dirs:
                shape_path = os.path.join(smoothness_path, shape_dir)
                if not os.path.isdir(shape_path):
                    continue

                shape_id = extract_shape_id(shape_dir)
                images = [os.path.join(shape_path, img) for img in os.listdir(shape_path) if img.endswith('.png')]
                if images:
                    shape_groups.setdefault(shape_id, []).extend(images)

            image_structure[extrusion_level][smoothness_level] = shape_groups

    return image_structure

# ----------------------- Geometry / constraints ---------------------- #

def calculate_spherical_coords(position, object_center) -> Tuple[float, float]:
    """Return (theta, phi) for position relative to object_center."""
    if isinstance(position, (list, tuple)) and len(position) == 3:
        rel_pos = [position[i] - object_center[i] for i in range(3)]
    else:
        rel_pos = position - object_center

    x, y, z = rel_pos
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0

    theta = math.acos(max(min(z / r, 1.0), -1.0))  # [0, pi]
    phi = math.atan2(y, x)                          # [-pi, pi]
    if phi < 0:
        phi += 2 * math.pi                          # [0, 2pi)
    return theta, phi


def spherical_distance(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Angular distance between two points on a sphere."""
    x1 = math.sin(theta1) * math.cos(phi1)
    y1 = math.sin(theta1) * math.sin(phi1)
    z1 = math.cos(theta1)
    x2 = math.sin(theta2) * math.cos(phi2)
    y2 = math.sin(theta2) * math.sin(phi2)
    z2 = math.cos(theta2)

    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    dot_product = max(min(dot_product, 1.0), -1.0)
    return math.acos(dot_product)


def is_in_sphere_restriction(theta: float, phi: float,
                             theta_center: float, phi_center: float,
                             delta_theta: float, delta_phi: float) -> bool:
    """Check if a (theta, phi) lies within a rectangular patch on the sphere."""
    theta_min = max(0.0, theta_center - delta_theta)
    theta_max = min(math.pi, theta_center + delta_theta)
    phi_min = phi_center - delta_phi
    phi_max = phi_center + delta_phi

    if not (theta_min <= theta <= theta_max):
        return False

    # Handle wrap-around for phi
    if phi_min < 0:
        return phi >= (phi_min + 2 * math.pi) or phi <= phi_max
    elif phi_max > 2 * math.pi:
        return phi >= phi_min or phi <= (phi_max - 2 * math.pi)
    else:
        return phi_min <= phi <= phi_max


def check_lat_lon_constraints(theta1: float, phi1: float,
                              theta2: float, phi2: float,
                              lat_thr: float, lon_thr: float) -> bool:
    """Latitude/longitude symmetry constraints."""
    if (abs(theta1 - theta2) < lat_thr or abs(math.pi - theta1 - theta2) < lat_thr):
        return False
    phi_diff = min(abs(phi1 - phi2), abs(2 * math.pi - abs(phi1 - phi2)))
    if phi_diff < lon_thr:
        return False
    return True

# ---------------------------- Core logic ----------------------------- #

def precompute_valid_pairs(
    image_structure: Dict[str, Dict[str, Dict[str, List[str]]]],
    camera_base_path: str,
    image_base_path: str,
    *,
    min_angle: float = math.radians(35),
    lat_threshold: float = math.radians(5),
    lon_threshold: float = math.radians(5),
    theta_center: float = math.pi / 2,
    phi_center: float = math.pi,
    delta_theta: float = math.pi / 2.3,
    delta_phi: float = math.pi / 2.3,
    disable_sphere_restriction: bool = False,
    max_pairs_per_object: int | None = None,
) -> Tuple[Dict[Tuple[str, str, str], List[Tuple[str, str]]], List[str], Dict[str, int | float]]:
    """
    Return (valid_pairs, unique_images, stats)
    valid_pairs: {(extrusion, smoothness, shape_id): [(img1, img2), ...], ...}
    unique_images: list of images present in valid_pairs
    stats: summary numbers
    """

    valid_pairs: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}

    # Count total objects for progress bar
    total_objects = sum(len(image_structure[ext][sm])
                        for ext in image_structure
                        for sm in image_structure[ext])

    print(f"\nPrecomputing valid pairs for {total_objects} objects...")
    pbar = tqdm(total=total_objects, desc="Computing valid pairs")

    objects_with_pairs = 0
    objects_without_pairs = 0
    total_pairs = 0

    for extrusion_level in sorted(image_structure.keys()):
        for smoothness_level in sorted(image_structure[extrusion_level].keys()):
            shape_groups = image_structure[extrusion_level][smoothness_level]

            for shape_id in sorted(shape_groups.keys()):
                pbar.set_description(f"ext_{extrusion_level}/smooth_{smoothness_level}/shape_{shape_id}")

                all_images = shape_groups[shape_id]

                # Collect camera data
                image_camera_data: List[Tuple[str, float, float]] = []
                for img_path in all_images:
                    camera_path = get_camera_info_path(img_path, camera_base_path, image_base_path)
                    if not os.path.exists(camera_path):
                        continue
                    cam_info = load_camera_info(camera_path)
                    if cam_info is None:
                        continue

                    position = cam_info.get('position', [0, 0, 0])
                    object_center = cam_info.get('object_center', [0, 0, 0])
                    theta, phi = calculate_spherical_coords(position, object_center)

                    if disable_sphere_restriction or is_in_sphere_restriction(theta, phi, theta_center, phi_center, delta_theta, delta_phi):
                        image_camera_data.append((img_path, theta, phi))

                # Compute valid pairs for this object
                object_valid_pairs: List[Tuple[str, str]] = []
                n = len(image_camera_data)
                for i in range(n):
                    img1, t1, p1 = image_camera_data[i]
                    for j in range(i + 1, n):
                        img2, t2, p2 = image_camera_data[j]

                        if spherical_distance(t1, p1, t2, p2) < min_angle:
                            continue
                        if not check_lat_lon_constraints(t1, p1, t2, p2, lat_threshold, lon_threshold):
                            continue
                        object_valid_pairs.append((img1, img2))

                # Subsample if needed BEFORE touching global unique set
                if max_pairs_per_object and len(object_valid_pairs) > max_pairs_per_object:
                    object_valid_pairs = random.sample(object_valid_pairs, max_pairs_per_object)

                key = (extrusion_level, smoothness_level, shape_id)
                if object_valid_pairs:
                    valid_pairs[key] = object_valid_pairs
                    objects_with_pairs += 1
                    total_pairs += len(object_valid_pairs)
                else:
                    objects_without_pairs += 1

                pbar.update(1)

    pbar.close()

    # Build unique image set only from kept pairs
    unique_images_set: Set[str] = set()
    for pairs in valid_pairs.values():
        for a, b in pairs:
            unique_images_set.add(a)
            unique_images_set.add(b)

    unique_images = sorted(unique_images_set)

    stats = {
        'total_pairs': total_pairs,
        'objects_with_pairs': objects_with_pairs,
        'objects_without_pairs': objects_without_pairs,
        'total_unique_images': len(unique_images),
        'average_pairs_per_object': (total_pairs / objects_with_pairs) if objects_with_pairs else 0.0
    }

    print("\nPrecomputation complete!")
    print(f"Objects with valid pairs: {objects_with_pairs}")
    print(f"Objects without valid pairs: {objects_without_pairs}")
    print(f"Total valid pairs: {total_pairs}")
    print(f"Total unique images to extract features from: {len(unique_images)}")
    if objects_with_pairs:
        print(f"Average pairs per object: {stats['average_pairs_per_object']:.1f}")

    return valid_pairs, unique_images, stats

# ------------------------------ Main --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Precompute valid image pairs for all objects')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path to the image directory structure')
    parser.add_argument('--camera_base_path', type=str, required=True,
                        help='Base path to the camera info directory structure')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for precomputed pairs (defaults to base_path)')

    # Camera constraint parameters
    parser.add_argument('--min_angle', type=float, default=35.0,
                        help='Minimum angular separation between cameras in degrees (default: 35)')
    parser.add_argument('--lat_threshold', type=float, default=5.0,
                        help='Latitude threshold for symmetry check in degrees (default: 5)')
    parser.add_argument('--lon_threshold', type=float, default=5.0,
                        help='Longitude threshold for symmetry check in degrees (default: 5)')
    parser.add_argument('--theta_center', type=float, default=math.pi / 2,
                        help='Theta center for sphere restriction in radians (default: PI/2)')
    parser.add_argument('--phi_center', type=float, default=math.pi,
                        help='Phi center for sphere restriction in radians (default: PI)')
    parser.add_argument('--delta_theta', type=float, default=math.pi / 2.3,
                        help='Delta theta for sphere restriction in radians (default: PI/2.3)')
    parser.add_argument('--delta_phi', type=float, default=math.pi / 2.3,
                        help='Delta phi for sphere restriction in radians (default: PI/2.3)')
    parser.add_argument('--disable_sphere_restriction', action='store_true',
                        help='Disable sphere restriction (use all camera positions)')
    parser.add_argument('--max_pairs_per_object', type=int, default=50,
                        help='Maximum number of valid pairs to keep per object (default: 50)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for subsampling')

    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_path is None:
        args.output_path = args.base_path

    output_dir = os.path.join(args.output_path, 'precomputed_pairs')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse image dirs
    image_structure = get_image_structure(args.base_path)

    # 2. Compute pairs
    valid_pairs, unique_images, stats = precompute_valid_pairs(
        image_structure,
        args.camera_base_path,
        args.base_path,
        min_angle=math.radians(args.min_angle),
        lat_threshold=math.radians(args.lat_threshold),
        lon_threshold=math.radians(args.lon_threshold),
        theta_center=args.theta_center,
        phi_center=args.phi_center,
        delta_theta=args.delta_theta,
        delta_phi=args.delta_phi,
        disable_sphere_restriction=args.disable_sphere_restriction,
        max_pairs_per_object=args.max_pairs_per_object,
    )

    # 3. Save outputs
    pairs_file = os.path.join(output_dir, 'valid_pairs.pkl')
    with open(pairs_file, 'wb') as f:
        pickle.dump(valid_pairs, f)
    print(f"\nValid pairs saved to: {pairs_file}")

    images_file = os.path.join(output_dir, 'unique_images.pkl')
    with open(images_file, 'wb') as f:
        pickle.dump(unique_images, f)
    print(f"Unique images list saved to: {images_file}")

    metadata = {
        'camera_constraints': {
            'min_angle': args.min_angle,
            'lat_threshold': args.lat_threshold,
            'lon_threshold': args.lon_threshold,
            'theta_center': args.theta_center,
            'phi_center': args.phi_center,
            'delta_theta': args.delta_theta,
            'delta_phi': args.delta_phi,
            'disable_sphere_restriction': args.disable_sphere_restriction
        },
        'stats': stats,
        'seed': args.seed,
        'max_pairs_per_object': args.max_pairs_per_object
    }
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == '__main__':
    main()
