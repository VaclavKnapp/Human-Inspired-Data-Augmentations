#!/usr/bin/env python3
"""
HIDA Dataset Generator
======================
Generates a unified CSV dataset with 300K diverse triplets:
- 50K Shapegen + black background
- 50K Shapegen + random background  
- 50K Shapegen + white background
- 50K Primigen + black background
- 50K Primigen + random background
- 50K Primigen + white background

Usage:
    python generate_hida_dataset.py --output hida_dataset.csv
"""

import argparse
import csv
import math
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

MIN_ANGLE = math.radians(30)
LAT_THRESHOLD = math.radians(3)
LON_THRESHOLD = math.radians(3)
THETA_CENTER = math.pi / 2
PHI_CENTER = math.pi
DELTA_THETA = math.pi / 2.7
DELTA_PHI = math.pi / 2.7


camera_info_cache: Dict[str, List[Dict[str, Any]]] = {}



def calculate_spherical_coords(position, object_center):

    if isinstance(position, (list, tuple)) and len(position) == 3:
        rel_pos = [position[i] - object_center[i] for i in range(3)]
    else:
        rel_pos = position - object_center
    
    x, y, z = rel_pos[0], rel_pos[1], rel_pos[2]
    r = math.sqrt(x*x + y*y + z*z)
    
    if r == 0:
        return 0.0, 0.0
    
    theta = math.acos(max(min(z / r, 1.0), -1.0))
    phi = math.atan2(y, x)
    if phi < 0:
        phi += 2 * math.pi
    
    return theta, phi

def load_camera_info(pkl_path: str):
    """Load camera info from pickle file"""
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Could not load camera info {pkl_path}: {e}")
        return None

def spherical_distance(theta1: float, phi1: float, theta2: float, phi2: float):
    """Great‑circle distance on a unit sphere."""
    x1 = math.sin(theta1) * math.cos(phi1)
    y1 = math.sin(theta1) * math.sin(phi1)
    z1 = math.cos(theta1)
    x2 = math.sin(theta2) * math.cos(phi2)
    y2 = math.sin(theta2) * math.sin(phi2)
    z2 = math.cos(theta2)
    cos_angle = max(min(x1 * x2 + y1 * y2 + z1 * z2, 1.0), -1.0)
    return math.acos(cos_angle)

def is_in_sphere_restriction(theta: float, phi: float):
    """Check if viewpoint is within sphere restriction"""
    theta_min = max(0, THETA_CENTER - DELTA_THETA)
    theta_max = min(math.pi, THETA_CENTER + DELTA_THETA)
    phi_min = PHI_CENTER - DELTA_PHI
    phi_max = PHI_CENTER + DELTA_PHI

    if not (theta_min <= theta <= theta_max):
        return False

    if phi_min < 0:
        return phi >= (phi_min + 2 * math.pi) or phi <= phi_max
    if phi_max > 2 * math.pi:
        return phi >= phi_min or phi <= (phi_max - 2 * math.pi)
    return phi_min <= phi <= phi_max

def angular_ok(t: float, p: float, chosen: List[Tuple[float, float]]):
    """Check separation + lat/lon symmetry."""
    for ct, cp in chosen:
        if spherical_distance(t, p, ct, cp) < MIN_ANGLE:
            return False
        if abs(t - ct) < LAT_THRESHOLD or abs(math.pi - t - ct) < LAT_THRESHOLD:
            return False
        pdiff = min(abs(p - cp), abs(2 * math.pi - abs(p - cp)))
        if pdiff < LON_THRESHOLD:
            return False
    return True

def camera_candidates(dir_path: str, camera_root: str):
    """Return list[(img_name, θ, φ)] or [] if no camera files."""
    if dir_path in camera_info_cache:
        cams = camera_info_cache[dir_path]
    else:
        cams: List[Dict] = []
        
        

        if "images_black_bg" in dir_path:

            rel_path = dir_path.split("images_black_bg/")[-1]
            camdir = os.path.join(camera_root, rel_path)
        elif "images_random_bg" in dir_path:

            rel_path = dir_path.split("images_random_bg/")[-1]
            camdir = os.path.join(camera_root, rel_path)
        elif "images_white_bg" in dir_path:
            # Handle white background paths
            rel_path = dir_path.split("images_white_bg/")[-1]
            camdir = os.path.join(camera_root, rel_path)
        else:

            parts = dir_path.split("/")
            dataset_idx = -1
            for i, part in enumerate(parts):
                if part in ["images_black_bg", "images_random_bg", "images_white_bg"]:
                    dataset_idx = i
                    break
            
            if dataset_idx != -1:
                rel_path = "/".join(parts[dataset_idx+1:])
                camdir = os.path.join(camera_root, rel_path)
            else:
                camdir = os.path.join(dir_path)
        
        if os.path.isdir(camdir):
            for fname in sorted(p for p in os.listdir(camdir) if p.endswith(".pkl")):
                cam_info = load_camera_info(os.path.join(camdir, fname))
                if cam_info is not None:
                    position = cam_info.get('position', [0, 0, 0])
                    object_center = cam_info.get('object_center', [0, 0, 0])
                    theta, phi = calculate_spherical_coords(position, object_center)
                    cam_info['theta'] = theta
                    cam_info['phi'] = phi
                cams.append(cam_info)
        else:
            print(f"[DEBUG] Camera dir not found: {camdir}")
            
        camera_info_cache[dir_path] = cams
    
    imgs = sorted(p for p in os.listdir(dir_path) if p.endswith(".png"))
    out = []
    for img in imgs:
        try:
            idx = int(img.split(".")[0])
            if idx < len(cams) and cams[idx] is not None:
                d = cams[idx]
                out.append((img, d["theta"], d["phi"]))
        except Exception:
            pass
    return out

def select_viewpoints(dir_path: str, camera_root: str, k: int = 2):
    """Pick ≤k images under camera constraints, else random fallback."""
    if not camera_root:
        # No camera constraints, select randomly
        imgs = [f for f in os.listdir(dir_path) if f.endswith(".png")]
        return random.sample(imgs, min(k, len(imgs)))
    
    cands = camera_candidates(dir_path, camera_root)
    if len(cands) < k:
        imgs = [f for f in os.listdir(dir_path) if f.endswith(".png")]
        return random.sample(imgs, min(k, len(imgs)))

    selected: List[str] = []
    positions: List[Tuple[float, float]] = []
    attempts = 0
    while len(selected) < k and attempts < 1000:
        attempts += 1
        img, th, ph = random.choice(cands)
        if img in selected:
            continue
        if is_in_sphere_restriction(th, ph) and angular_ok(th, ph, positions):
            selected.append(img)
            positions.append((th, ph))
    
    if len(selected) < k:
        remaining = [f for f in os.listdir(dir_path) if f.endswith(".png") and f not in selected]
        selected += random.sample(remaining, min(k - len(selected), len(remaining)))
    return selected



def load_similarity_data(similarity_file: str):
    """Load similarity data from pickle file"""
    try:
        with open(similarity_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded similarity data for {len(data)} extrusion levels")
        return data
    except Exception as e:
        print(f"Error loading similarity data: {e}")
        return {}

def get_shapegen_structure(base_path: str):
    """Get Shapegen directory structure: extrusion -> smoothness -> shape_id -> path
    Maps similarity data keys (like 'shape_012') to directory paths"""
    structure = {}
    if not os.path.isdir(base_path):
        return structure
    
    for ext_dir in sorted(d for d in os.listdir(base_path) if d.startswith("extrusions_")):
        ext_path = os.path.join(base_path, ext_dir)
        if not os.path.isdir(ext_path):
            continue
        
        ext_level = ext_dir.split("_")[1]
        structure[ext_level] = {}
        
        for smooth_dir in sorted(d for d in os.listdir(ext_path) if d.startswith("smoothness_")):
            smooth_path = os.path.join(ext_path, smooth_dir)
            if not os.path.isdir(smooth_path):
                continue
            
            smooth_level = smooth_dir.split("_")[1]
            structure[ext_level][smooth_level] = {}
            
            for shape_dir in sorted(d for d in os.listdir(smooth_path) if d.startswith("shape_")):
                shape_path = os.path.join(smooth_path, shape_dir)
                if os.path.isdir(shape_path):
                    # Extract shape ID from directory name like "shape_smoothness_5_012"
                    # to match similarity data keys like "shape_012"
                    parts = shape_dir.split("_")
                    if len(parts) >= 4:
                        shape_id = parts[-1]  # Get "012" 
                        shape_key = f"shape_{shape_id}"  # Create "shape_012"
                        structure[ext_level][smooth_level][shape_key] = shape_path
    
    return structure

def calculate_object_similarity_bins(extrusion_key: str, smoothness_key: str, target_object: str, similarity_data: Dict):
    """Calculate 3 similarity bins for above-mean objects."""
    if (extrusion_key not in similarity_data or 
        smoothness_key not in similarity_data[extrusion_key]):
        return [], [], [], {}
    
    similarities = similarity_data[extrusion_key][smoothness_key]
    
    if target_object not in similarities:
        return [], [], [], {}
    
    # Get similarities to all other objects
    object_similarities = []
    other_objects = []
    
    for other_obj, sim_value in similarities[target_object].items():
        if other_obj != target_object:
            object_similarities.append(sim_value)
            other_objects.append(other_obj)
    
    if len(object_similarities) == 0:
        return [], [], [], {}
    
    # Calculate mean and max
    mean_sim = np.mean(object_similarities)
    max_sim = np.max(object_similarities)
    
    # Create 3 equal-width bins from mean to max (ABOVE MEAN only)
    bin_width = (max_sim - mean_sim) / 3
    bin_edges = [
        mean_sim,
        mean_sim + bin_width,
        mean_sim + 2 * bin_width,
        max_sim
    ]
    
    # Categorize objects into above-mean bins
    bin1, bin2, bin3 = [], [], []
    
    for obj, sim in zip(other_objects, object_similarities):
        if bin_edges[0] <= sim < bin_edges[1]:
            bin1.append((obj, sim))
        elif bin_edges[1] <= sim < bin_edges[2]:
            bin2.append((obj, sim))
        elif bin_edges[2] <= sim <= bin_edges[3]:
            bin3.append((obj, sim))
    
    # Sort by similarity within each bin
    bin1.sort(key=lambda x: x[1], reverse=True)
    bin2.sort(key=lambda x: x[1], reverse=True)
    bin3.sort(key=lambda x: x[1], reverse=True)
    
    bin_info = {
        'mean': mean_sim,
        'max': max_sim,
        'bin_edges': bin_edges,
        'counts': [len(bin1), len(bin2), len(bin3)]
    }
    
    return bin1, bin2, bin3, bin_info

def generate_shapegen_triplet(base_path: str, camera_path: str, structure: Dict, similarity_data: Dict, bin_type: str,
                             allowed_objects: List = None):
    """Generate one Shapegen triplet based on similarity bins."""
    # Select random extrusion and smoothness
    extrusion_levels = list(structure.keys())
    if not extrusion_levels:
        return None

    extrusion = random.choice(extrusion_levels)
    smoothness_levels = list(structure[extrusion].keys())
    if not smoothness_levels:
        return None

    smoothness = random.choice(smoothness_levels)

    extrusion_key = f"extrusion_{extrusion}"
    smoothness_key = f"smoothness_{smoothness}"
    
    # Check if similarity data exists for this combination
    if (extrusion_key not in similarity_data or 
        smoothness_key not in similarity_data[extrusion_key]):
        print(f"[DEBUG] Missing similarity data for {extrusion_key}/{smoothness_key}")
        print(f"[DEBUG] Available extrusion keys: {list(similarity_data.keys())}")
        if extrusion_key in similarity_data:
            print(f"[DEBUG] Available smoothness keys for {extrusion_key}: {list(similarity_data[extrusion_key].keys())}")
        return None
    
    # Select focal object from similarity data keys
    available_shapes = list(structure[extrusion][smoothness].keys())
    similarities = similarity_data[extrusion_key][smoothness_key]

    # Find shapes that exist in both structure and similarity data
    valid_shapes = [shape for shape in available_shapes if shape in similarities]

    # Filter by allowed_objects if manifest is provided
    if allowed_objects is not None:
        # allowed_objects contains tuples of (extrusion, smoothness, shape_id)
        allowed_shape_ids = set()
        for obj_tuple in allowed_objects:
            if len(obj_tuple) == 3 and obj_tuple[0] == extrusion and obj_tuple[1] == smoothness:
                allowed_shape_ids.add(f"shape_{obj_tuple[2]}")

        if allowed_shape_ids:
            valid_shapes = [s for s in valid_shapes if s in allowed_shape_ids]

    if len(valid_shapes) < 2:
        print(f"[DEBUG] Not enough valid shapes for {extrusion_key}/{smoothness_key}: {len(valid_shapes)}")
        print(f"[DEBUG] Available shapes in structure: {available_shapes[:5]}...")
        print(f"[DEBUG] Available shapes in similarity: {list(similarities.keys())[:5]}...")
        return None

    focal_shape = random.choice(valid_shapes)
    
    # Get similarity bins
    bin1, bin2, bin3, bin_info = calculate_object_similarity_bins(
        extrusion_key, smoothness_key, focal_shape, similarity_data
    )
    
    bins = [bin1, bin2, bin3]
    bin_index = ['low', 'medium', 'high'].index(bin_type)
    
    # Check if the requested bin has enough objects
    if len(bins[bin_index]) < 1:
        # Try other bins if requested bin is empty
        for i, bin_objects in enumerate(bins):
            if len(bin_objects) >= 1:
                bin_index = i
                bin_type = ['low', 'medium', 'high'][i]
                break
        else:
            return None
    
    # Get focal object images (A and A')
    focal_dir = structure[extrusion][smoothness][focal_shape]
    focal_imgs = select_viewpoints(focal_dir, camera_path, k=2)
    if len(focal_imgs) < 2:
        print(f"[DEBUG] Not enough focal images in {focal_dir}")
        return None

    img_a, img_a_prime = random.sample(focal_imgs, 2)
    img_a_path = os.path.join(focal_dir, img_a)
    img_a_prime_path = os.path.join(focal_dir, img_a_prime)

    # Get similar object image (B)
    similar_obj, similarity_score = random.choice(bins[bin_index])
    if similar_obj not in structure[extrusion][smoothness]:
        print(f"[DEBUG] Similar object {similar_obj} not in structure")
        return None

    similar_dir = structure[extrusion][smoothness][similar_obj]
    similar_imgs = select_viewpoints(similar_dir, camera_path, k=1)
    if not similar_imgs:
        print(f"[DEBUG] No similar images in {similar_dir}")
        return None
    
    img_b = random.choice(similar_imgs)
    img_b_path = os.path.join(similar_dir, img_b)
    
    # Verify all paths exist
    if not all(os.path.exists(p) for p in [img_a_path, img_a_prime_path, img_b_path]):
        print(f"[DEBUG] Some image paths don't exist")
        return None
    
    # Create condition string - just the similarity value
    condition = f"{similarity_score:.3f}"
    
    # Create descriptive names with full path context
    name_a = f"extrusion_{extrusion}/smoothness_{smoothness}/{focal_shape}"
    name_b = f"extrusion_{extrusion}/smoothness_{smoothness}/{similar_obj}"

    # Removed verbose debug print - progress is shown in generate_shapegen_triplets()
    return {
        'A': img_a_path,
        'A_prime': img_a_prime_path,
        'B': img_b_path,
        'condition': condition,
        'name_a': name_a,
        'name_b': name_b,
        'similarity_score': similarity_score
    }

def generate_shapegen_triplets(base_path: str, similarity_file: str, camera_path: str,
                              bg_type: str, ratios: List[float], n_triplets: int,
                              manifest_objects: Dict = None):
    """Generate n_triplets for Shapegen dataset."""
    print(f"Generating {n_triplets} Shapegen triplets ({bg_type} background)...")

    # Load similarity data and structure
    similarity_data = load_similarity_data(similarity_file)
    if not similarity_data:
        print(f"Failed to load similarity data from {similarity_file}")
        return []

    structure = get_shapegen_structure(base_path)
    if not structure:
        print(f"No Shapegen structure found in {base_path}")
        return []
    
    print(f"  Found {len(structure)} extrusion levels")
    for ext, smoothness_dict in structure.items():
        for smooth, shapes in smoothness_dict.items():
            print(f"    Extrusion {ext}, Smoothness {smooth}: {len(shapes)} shapes")
    
    print(f"  Similarity data has {len(similarity_data)} extrusion keys")
    for ext_key, smooth_dict in similarity_data.items():
        print(f"    {ext_key}: {len(smooth_dict)} smoothness levels")
        for smooth_key, similarities in smooth_dict.items():
            print(f"      {smooth_key}: {len(similarities)} objects")
    
    triplets = []
    bin_types = ['low', 'medium', 'high']
    bin_counts = [int(n_triplets * r) for r in ratios]
    
    # Ensure we generate exactly n_triplets
    total_assigned = sum(bin_counts)
    if total_assigned < n_triplets:
        bin_counts[-1] += n_triplets - total_assigned
    
    for bin_type, target_count in zip(bin_types, bin_counts):
        print(f"  Generating {target_count} {bin_type} similarity triplets...")

        # Get allowed objects for this bin_type if manifest is provided
        allowed_objects = None
        if manifest_objects is not None and bg_type in manifest_objects:
            allowed_objects = manifest_objects[bg_type].get(bin_type, None)

        attempts = 0
        generated = 0
        max_attempts = target_count * 20  # Increased attempts
        consecutive_failures = 0

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            triplet = generate_shapegen_triplet(base_path, camera_path, structure, similarity_data, bin_type,
                                               allowed_objects)
            
            if triplet:
                triplet.update({
                    'trial': len(triplets) + 1,
                    'bg': bg_type,
                    'dataset': 'SHAPEGEN'
                })
                triplets.append(triplet)
                generated += 1
                consecutive_failures = 0

                if generated % 1000 == 0:
                    print(f"    Generated {generated}/{target_count} {bin_type} similarity triplets")
            else:
                consecutive_failures += 1
                if consecutive_failures > 1000:
                    print(f"    Warning: Too many consecutive failures, may have issues with data")
                    break
    
    print(f"Generated {len(triplets)} Shapegen triplets")
    return triplets

# ──────────────────────────── Primigen Functions ────────────────────────────

def get_primigen_structure(base_path: str):
    """Get Primigen directory structure: n -> config -> warp -> place -> path"""
    structure = {}
    if not os.path.isdir(base_path):
        return structure
    
    for n_dir in sorted(d for d in os.listdir(base_path) if d.lower().startswith("n")):
        n_path = os.path.join(base_path, n_dir)
        if not os.path.isdir(n_path):
            continue
        
        n_level = n_dir
        structure[n_level] = {}
        
        for cfg in sorted(d for d in os.listdir(n_path) if d.startswith("config_")):
            cfg_path = os.path.join(n_path, cfg)
            if not os.path.isdir(cfg_path):
                continue
            
            structure[n_level][cfg] = {}
            
            for warp in sorted(d for d in os.listdir(cfg_path) if d.startswith("warp_")):
                warp_path = os.path.join(cfg_path, warp)
                if not os.path.isdir(warp_path):
                    continue
                
                structure[n_level][cfg][warp] = {}
                
                for place in sorted(d for d in os.listdir(warp_path) if d.startswith("place_")):
                    place_path = os.path.join(warp_path, place)
                    if os.path.isdir(place_path):
                        structure[n_level][cfg][warp][place] = place_path
    
    return structure

def generate_primigen_triplet(structure: Dict, camera_path: str, condition_type: str, n_level: str = None,
                              allowed_objects: List = None):
    """Generate one Primigen triplet for the specified condition type."""
    if not structure:
        return None

    # Select n_level if not specified
    if n_level is None:
        n_level = random.choice(list(structure.keys()))

    if n_level not in structure:
        return None
    
    if condition_type == "place":
        # Same warp, different places
        cfg = random.choice(list(structure[n_level].keys()))
        warp = random.choice(list(structure[n_level][cfg].keys()))
        places = list(structure[n_level][cfg][warp].keys())
        
        if len(places) < 2:
            return None
        
        place_a, place_b = random.sample(places, 2)
        dir_a = structure[n_level][cfg][warp][place_a]
        dir_b = structure[n_level][cfg][warp][place_b]
        
        name_a = f"{n_level}_{cfg}_{warp}_{place_a}"
        name_b = f"{n_level}_{cfg}_{warp}_{place_b}"
        
    elif condition_type == "warp":
        # Same config, different warps
        cfg = random.choice(list(structure[n_level].keys()))
        warps = list(structure[n_level][cfg].keys())
        
        if len(warps) < 2:
            return None
        
        warp_a, warp_b = random.sample(warps, 2)
        place_a = random.choice(list(structure[n_level][cfg][warp_a].keys()))
        place_b = random.choice(list(structure[n_level][cfg][warp_b].keys()))
        
        dir_a = structure[n_level][cfg][warp_a][place_a]
        dir_b = structure[n_level][cfg][warp_b][place_b]
        
        name_a = f"{n_level}_{cfg}_{warp_a}_{place_a}"
        name_b = f"{n_level}_{cfg}_{warp_b}_{place_b}"
        
    elif condition_type == "config":
        # Different configs
        cfgs = list(structure[n_level].keys())
        
        if len(cfgs) < 2:
            return None
        
        cfg_a, cfg_b = random.sample(cfgs, 2)
        warp_a = random.choice(list(structure[n_level][cfg_a].keys()))
        place_a = random.choice(list(structure[n_level][cfg_a][warp_a].keys()))
        warp_b = random.choice(list(structure[n_level][cfg_b].keys()))
        place_b = random.choice(list(structure[n_level][cfg_b][warp_b].keys()))
        
        dir_a = structure[n_level][cfg_a][warp_a][place_a]
        dir_b = structure[n_level][cfg_b][warp_b][place_b]
        
        name_a = f"{n_level}_{cfg_a}_{warp_a}_{place_a}"
        name_b = f"{n_level}_{cfg_b}_{warp_b}_{place_b}"
    else:
        return None
    
    # Select viewpoints
    imgs_a = select_viewpoints(dir_a, camera_path, k=2)
    if len(imgs_a) < 2:
        return None

    imgs_b = select_viewpoints(dir_b, camera_path, k=1)
    if not imgs_b:
        return None
    
    img_a, img_a_prime = random.sample(imgs_a, 2)
    img_b = random.choice(imgs_b)
    
    img_a_path = os.path.join(dir_a, img_a)
    img_a_prime_path = os.path.join(dir_a, img_a_prime)
    img_b_path = os.path.join(dir_b, img_b)
    
    # Verify all paths exist
    if not all(os.path.exists(p) for p in [img_a_path, img_a_prime_path, img_b_path]):
        return None
    
    condition = f"{n_level}_{condition_type}"
    
    return {
        'A': img_a_path,
        'A_prime': img_a_prime_path,
        'B': img_b_path,
        'condition': condition,
        'name_a': name_a,
        'name_b': name_b
    }

def generate_primigen_triplets(base_path: str, camera_path: str, bg_type: str,
                              condition_ratios: List[float], n_ratios: List[float], n_triplets: int,
                              manifest_objects: Dict = None):
    """Generate n_triplets for Primigen dataset."""
    print(f"Generating {n_triplets} Primigen triplets ({bg_type} background)...")

    structure = get_primigen_structure(base_path)
    if not structure:
        print(f"No Primigen structure found in {base_path}")
        return []

    triplets = []
    condition_types = ['place', 'warp', 'config']
    n_levels = list(structure.keys())

    # Calculate target counts
    condition_counts = [int(n_triplets * r) for r in condition_ratios]
    total_assigned = sum(condition_counts)
    if total_assigned < n_triplets:
        condition_counts[-1] += n_triplets - total_assigned

    for condition_type, target_count in zip(condition_types, condition_counts):
        print(f"  Generating {target_count} {condition_type} triplets...")

        # Distribute across n_levels
        n_counts = [int(target_count * r) for r in n_ratios]
        total_n_assigned = sum(n_counts)
        if total_n_assigned < target_count:
            n_counts[-1] += target_count - total_n_assigned

        for n_level, n_target in zip(n_levels, n_counts):
            # Get allowed objects for this condition_type if manifest is provided
            allowed_objects = None
            if manifest_objects is not None and bg_type in manifest_objects:
                allowed_objects = manifest_objects[bg_type].get(condition_type, None)

            attempts = 0
            generated = 0
            max_attempts = n_target * 10

            while generated < n_target and attempts < max_attempts:
                attempts += 1

                triplet = generate_primigen_triplet(structure, camera_path, condition_type, n_level, allowed_objects)

                if triplet:
                    triplet.update({
                        'trial': len(triplets) + 1,
                        'bg': bg_type,
                        'dataset': 'PRIMIGEN'
                    })
                    triplets.append(triplet)
                    generated += 1

                    if generated % 1000 == 0:
                        print(f"    Generated {generated}/{n_target} for {n_level}")

    print(f"Generated {len(triplets)} Primigen triplets")
    return triplets

# ──────────────────────────── Main Generation Function ────────────────────────────

def save_to_csv(triplets: List[Dict], output_path: str):
    """Save triplets to CSV file."""
    print(f"Saving {len(triplets)} triplets to {output_path}...")
    
    headers = ['Trial', 'BG', 'A', 'A_prime', 'B', 'DATASET', 'CONDITION', 'NAME_A', 'NAME_B']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for i, triplet in enumerate(triplets, 1):
            writer.writerow([
                i,  # Sequential trial number
                triplet['bg'],
                triplet['A'],
                triplet['A_prime'],
                triplet['B'],
                triplet['dataset'],
                triplet['condition'],
                triplet['name_a'],
                triplet['name_b']
            ])
    
    print(f"Saved to {output_path}")

def generate_hida_dataset(
    shapegen_black_path: str,
    shapegen_random_path: str,
    shapegen_white_path: str,
    shapegen_black_sim_file: str,
    shapegen_random_sim_file: str,
    shapegen_white_sim_file: str,
    shapegen_camera_path: str,
    primigen_black_path: str,
    primigen_random_path: str,
    primigen_white_path: str,
    primigen_camera_path: str,
    output_csv_path: str,
    shapegen_ratios: List[float] = [0.20, 0.45, 0.35],  # low, medium, high similarity
    primigen_ratios: List[float] = [0.55, 0.27, 0.18],  # place, warp, config
    n_ratios: List[float] = [0.43, 0.42, 0.15],  # n2, n3, n4
    triplets_per_combo: int = 20000,
    load_manifest_path: Optional[str] = None
):
    """Generate the complete HIDA dataset."""
    print("🚀 Generating HIDA Dataset...")
    print(f"Target: {6 * triplets_per_combo:,} total triplets")
    print(f"Shapegen similarity ratios: {shapegen_ratios}")
    print(f"Primigen condition ratios: {primigen_ratios}")
    print(f"N-level ratios: {n_ratios}")
    print(f"Black background sim file: {os.path.basename(shapegen_black_sim_file)}")
    print(f"Random background sim file: {os.path.basename(shapegen_random_sim_file)}")
    print(f"White background sim file: {os.path.basename(shapegen_white_sim_file)}")

    # Load manifest if provided
    shapegen_manifest = None
    primigen_manifest = None
    if load_manifest_path and os.path.exists(load_manifest_path):
        print(f"📋 Loading object manifest from {load_manifest_path}")
        try:
            import json
            with open(load_manifest_path, 'r') as f:
                manifest_data = json.load(f)
            shapegen_manifest = manifest_data.get('shapegen', None)
            primigen_manifest = manifest_data.get('primigen', None)
            print(f"  Loaded manifest with {len(shapegen_manifest) if shapegen_manifest else 0} shapegen backgrounds")
            print(f"  Loaded manifest with {len(primigen_manifest) if primigen_manifest else 0} primigen backgrounds")
        except Exception as e:
            print(f"  Warning: Failed to load manifest: {e}")
            shapegen_manifest = None
            primigen_manifest = None

    all_triplets = []
    
    # Generate 50K for each combination
    try:
        # Shapegen + black background
        shapegen_black = generate_shapegen_triplets(
            shapegen_black_path, shapegen_black_sim_file, shapegen_camera_path,
            "BLACK", shapegen_ratios, triplets_per_combo, shapegen_manifest
        )
        all_triplets.extend(shapegen_black)

        # Shapegen + random background
        shapegen_random = generate_shapegen_triplets(
            shapegen_random_path, shapegen_random_sim_file, shapegen_camera_path,
            "RANDOM", shapegen_ratios, triplets_per_combo, shapegen_manifest
        )
        all_triplets.extend(shapegen_random)

        # Shapegen + white background
        shapegen_white = generate_shapegen_triplets(
            shapegen_white_path, shapegen_white_sim_file, shapegen_camera_path,
            "WHITE", shapegen_ratios, triplets_per_combo, shapegen_manifest
        )
        all_triplets.extend(shapegen_white)

        # Primigen + black background
        primigen_black = generate_primigen_triplets(
            primigen_black_path, primigen_camera_path, "BLACK",
            primigen_ratios, n_ratios, triplets_per_combo, primigen_manifest
        )
        all_triplets.extend(primigen_black)

        # Primigen + random background
        primigen_random = generate_primigen_triplets(
            primigen_random_path, primigen_camera_path, "RANDOM",
            primigen_ratios, n_ratios, triplets_per_combo, primigen_manifest
        )
        all_triplets.extend(primigen_random)

        # Primigen + white background
        primigen_white = generate_primigen_triplets(
            primigen_white_path, primigen_camera_path, "WHITE",
            primigen_ratios, n_ratios, triplets_per_combo, primigen_manifest
        )
        all_triplets.extend(primigen_white)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    print(f"\nGenerated {len(all_triplets):,} total triplets")
    
    # Shuffle and save
    print("Shuffling triplets...")
    random.shuffle(all_triplets)
    
    save_to_csv(all_triplets, output_csv_path)
    
    print("HIDA dataset generation complete!")

def main():
    parser = argparse.ArgumentParser(description='Generate HIDA Dataset')
    
    # Required paths
    parser.add_argument('--shapegen-black', required=True,
                       help='Path to Shapegen black background images')
    parser.add_argument('--shapegen-random', required=True,
                       help='Path to Shapegen random background images')
    parser.add_argument('--shapegen-white', required=True,
                       help='Path to Shapegen white background images')
    parser.add_argument('--shapegen-black-sim', required=True,
                       help='Path to Shapegen black background similarity file (.pkl)')
    parser.add_argument('--shapegen-random-sim', required=True,
                       help='Path to Shapegen random background similarity file (.pkl)')
    parser.add_argument('--shapegen-white-sim', required=True,
                       help='Path to Shapegen white background similarity file (.pkl)')
    parser.add_argument('--shapegen-camera', required=True,
                       help='Path to Shapegen camera info directory')
    parser.add_argument('--primigen-black', required=True,
                       help='Path to Primigen black background images')
    parser.add_argument('--primigen-random', required=True,
                       help='Path to Primigen random background images')
    parser.add_argument('--primigen-white', required=True,
                       help='Path to Primigen white background images')
    parser.add_argument('--primigen-camera', required=True,
                       help='Path to Primigen camera info directory')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    
    # Optional ratios
    parser.add_argument('--shapegen-ratios', nargs=3, type=float, 
                       default=[0.25, 0.45, 0.30],
                       help='Ratios for low/medium/high similarity (default: 0.33 0.33 0.34)')
    parser.add_argument('--primigen-ratios', nargs=3, type=float,
                       default=[0.60, 0.25, 0.15], 
                       help='Ratios for place/warp/config (default: 0.33 0.33 0.34)')
    parser.add_argument('--n-ratios', nargs=3, type=float,
                       default=[0.40, 0.40, 0.20],
                       help='Ratios for n2/n3/n4 (default: 0.33 0.33 0.34)')
    parser.add_argument('--triplets-per-combo', type=int, default=20000,
                       help='Number of triplets per combination (default: 50000)')
    parser.add_argument('--seed', type=int, default=42)

    # Manifest control
    parser.add_argument('--load-object-manifest', type=str, default=None,
                       help='Path to object manifest JSON to load (restricts object pool)')

    args = parser.parse_args()
    
    # Validate ratio sums
    for ratios, name in [(args.shapegen_ratios, 'shapegen'), 
                        (args.primigen_ratios, 'primigen'),
                        (args.n_ratios, 'n-level')]:
        if abs(sum(ratios) - 1.0) > 0.01:
            print(f"Error: {name} ratios must sum to 1.0, got {sum(ratios)}")
            sys.exit(1)
    
    # Validate paths
    paths_to_check = [
        (args.shapegen_black, "Shapegen black"),
        (args.shapegen_random, "Shapegen random"),
        (args.shapegen_white, "Shapegen white"),
        (args.shapegen_black_sim, "Shapegen black similarity file"),
        (args.shapegen_random_sim, "Shapegen random similarity file"),
        (args.shapegen_white_sim, "Shapegen white similarity file"),
        (args.shapegen_camera, "Shapegen camera"),
        (args.primigen_black, "Primigen black"),
        (args.primigen_random, "Primigen random"),
        (args.primigen_white, "Primigen white"),
        (args.primigen_camera, "Primigen camera")
    ]
    
    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"Error: {name} path does not exist: {path}")
            sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(args.seed)

    generate_hida_dataset(
        args.shapegen_black,
        args.shapegen_random,
        args.shapegen_white,
        args.shapegen_black_sim,
        args.shapegen_random_sim,
        args.shapegen_white_sim,
        args.shapegen_camera,
        args.primigen_black,
        args.primigen_random,
        args.primigen_white,
        args.primigen_camera,
        args.output,
        args.shapegen_ratios,
        args.primigen_ratios,
        args.n_ratios,
        args.triplets_per_combo,
        args.load_object_manifest
    )

if __name__ == "__main__":
    main()