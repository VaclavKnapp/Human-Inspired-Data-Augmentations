#!/usr/bin/env python3
"""
HIDA Dataset Generator with Objaverse and CO3D
===============================================
Generates a unified CSV dataset with diverse triplets from four datasets:
- Primigen (black, random, white backgrounds)
- Shapegen (black, random, white backgrounds)
- Objaverse (black, random, white backgrounds)
- CO3D (black, random, white backgrounds)

Usage:
    python create_csv_w_Objaverse_CO3D.py --output hida_dataset_complete.csv [args...]
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

def select_objaverse_viewpoints(dir_path: str, k: int = 2):
    """Select viewpoints for Objaverse using 360-degree rotation logic."""
    imgs = sorted([f for f in os.listdir(dir_path) if f.endswith(".png")])
    
    if len(imgs) < k:
        return imgs
    
    if k == 2:
        # Select A and A' to be opposite views in the 360° rotation
        img_a_name = random.choice(imgs)
        try:
            img_a_num = int(img_a_name.split('.')[0])
            total_images = len(imgs)
            
            # Calculate opposite view (180 degrees apart)
            # Since images represent a full 360° rotation, opposite view is at +half_rotation
            half_rotation = total_images // 2
            img_a_prime_num = (img_a_num + half_rotation) % total_images
            
            # Find the image with this number
            img_a_prime_name = f"{img_a_prime_num:03d}.png"
            
            # Check if A' exists in the directory
            if img_a_prime_name in imgs:
                return [img_a_name, img_a_prime_name]
            else:
                # If calculated A' doesn't exist, try adjacent images
                for offset in [1, -1, 2, -2]:
                    candidate_num = (img_a_prime_num + offset) % total_images
                    candidate_name = f"{candidate_num:03d}.png"
                    if candidate_name in imgs and candidate_name != img_a_name:
                        return [img_a_name, candidate_name]
                
                # Final fallback to random selection
                return random.sample(imgs, 2)
                
        except (ValueError, IndexError):
            # Fallback to random selection if naming doesn't match expected pattern
            return random.sample(imgs, 2)
    else:
        # For other k values, just select randomly
        return random.sample(imgs, k)


# ──────────────────────────── CO3D Functions ────────────────────────────

def select_co3d_viewpoints(dir_path: str, k: int = 2):
    """Select viewpoints for CO3D using frame rotation logic similar to Objaverse."""
    imgs = sorted([f for f in os.listdir(dir_path) if f.endswith(".png")])
    
    if len(imgs) < k:
        return imgs
    
    if k == 2:
        # Select A and A' to be maximally separated views
        img_a_name = random.choice(imgs)
        try:
            # Extract frame number from 'frame000001.png' format
            frame_num_str = img_a_name.replace('frame', '').replace('.png', '')
            img_a_num = int(frame_num_str)
            total_images = len(imgs)
            
            # Calculate opposite view (180 degrees apart conceptually)
            half_rotation = total_images // 2
            img_a_prime_num = img_a_num + half_rotation
            
            # Handle wraparound if needed
            if img_a_prime_num > total_images:
                img_a_prime_num = img_a_prime_num - total_images
            
            # Format frame number with leading zeros
            img_a_prime_name = f"frame{img_a_prime_num:06d}.png"
            
            # Check if A' exists in the directory
            if img_a_prime_name in imgs:
                return [img_a_name, img_a_prime_name]
            else:
                # If calculated A' doesn't exist, try adjacent frames
                for offset in [1, -1, 2, -2, 3, -3]:
                    candidate_num = img_a_prime_num + offset
                    if candidate_num < 1:
                        candidate_num = total_images + candidate_num
                    elif candidate_num > total_images:
                        candidate_num = candidate_num - total_images
                    
                    candidate_name = f"frame{candidate_num:06d}.png"
                    if candidate_name in imgs and candidate_name != img_a_name:
                        return [img_a_name, candidate_name]
                
                # Final fallback to random selection
                return random.sample(imgs, 2)
                
        except (ValueError, IndexError):
            # Fallback to random selection if naming doesn't match expected pattern
            return random.sample(imgs, 2)
    else:
        # For other k values, just select randomly
        return random.sample(imgs, k)

def get_co3d_structure(base_path: str):
    """Get CO3D directory structure: category -> sequence -> images_path"""
    structure = {}
    if not os.path.isdir(base_path):
        return structure
    
    for category_dir in sorted(os.listdir(base_path)):
        category_path = os.path.join(base_path, category_dir)
        if not os.path.isdir(category_path):
            continue
        
        structure[category_dir] = {}
        
        for sequence_dir in sorted(os.listdir(category_path)):
            sequence_path = os.path.join(category_path, sequence_dir)
            if not os.path.isdir(sequence_path):
                continue
            
            # Check for images subdirectory
            images_path = os.path.join(sequence_path, "images")
            if os.path.isdir(images_path):
                # Check if there are actually images
                imgs = [f for f in os.listdir(images_path) if f.endswith('.png')]
                if len(imgs) > 0:
                    structure[category_dir][sequence_dir] = images_path
    
    return structure

def calculate_co3d_similarity_bins(target_sequence: str, category: str, similarity_data: Dict):
    """Calculate 3 similarity bins for above-mean sequences within a category."""
    if category not in similarity_data:
        return [], [], [], {}
    
    if target_sequence not in similarity_data[category]:
        return [], [], [], {}
    
    # Get similarities to all other sequences in the same category
    sequence_similarities = []
    other_sequences = []
    
    for other_seq, sim_value in similarity_data[category][target_sequence].items():
        if other_seq != target_sequence:
            sequence_similarities.append(sim_value)
            other_sequences.append(other_seq)
    
    if len(sequence_similarities) == 0:
        return [], [], [], {}
    
    # Calculate mean and max
    mean_sim = np.mean(sequence_similarities)
    max_sim = np.max(sequence_similarities)
    
    # Create 3 equal-width bins from mean to max (ABOVE MEAN only)
    bin_width = (max_sim - mean_sim) / 3
    bin_edges = [
        mean_sim,
        mean_sim + bin_width,
        mean_sim + 2 * bin_width,
        max_sim
    ]
    
    # Categorize sequences into above-mean bins
    bin1, bin2, bin3 = [], [], []
    
    for seq, sim in zip(other_sequences, sequence_similarities):
        if bin_edges[0] <= sim < bin_edges[1]:
            bin1.append((seq, sim))
        elif bin_edges[1] <= sim < bin_edges[2]:
            bin2.append((seq, sim))
        elif bin_edges[2] <= sim <= bin_edges[3]:
            bin3.append((seq, sim))
    
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

def generate_co3d_triplet(structure: Dict, similarity_data: Dict, bin_type: str):
    """Generate one CO3D triplet based on similarity bins within same category."""
    # Select a random category that has data
    available_categories = [cat for cat in structure.keys() if cat in similarity_data]
    if not available_categories:
        return None
    
    category = random.choice(available_categories)
    
    # Get sequences in this category
    sequences = structure[category]
    available_sequences = [seq for seq in sequences.keys() if seq in similarity_data.get(category, {})]
    
    if len(available_sequences) < 2:
        return None
    
    # Select focal sequence
    focal_sequence = random.choice(available_sequences)
    
    # Get similarity bins
    bin1, bin2, bin3, _ = calculate_co3d_similarity_bins(
        focal_sequence, category, similarity_data
    )
    
    bins = [bin1, bin2, bin3]
    bin_index = ['low', 'medium', 'high'].index(bin_type)
    
    # Check if the requested bin has enough sequences
    if len(bins[bin_index]) < 1:
        # Try other bins if requested bin is empty
        for i, bin_sequences in enumerate(bins):
            if len(bin_sequences) >= 1:
                bin_index = i
                bin_type = ['low', 'medium', 'high'][i]
                break
        else:
            return None
    
    # Get focal sequence images (A and A') using CO3D-specific selection
    focal_dir = sequences[focal_sequence]
    focal_imgs = select_co3d_viewpoints(focal_dir, 2)
    if len(focal_imgs) < 2:
        return None
    
    img_a, img_a_prime = focal_imgs[0], focal_imgs[1]
    img_a_path = os.path.join(focal_dir, img_a)
    img_a_prime_path = os.path.join(focal_dir, img_a_prime)
    
    # Get similar sequence image (B)
    similar_seq, similarity_score = random.choice(bins[bin_index])
    if similar_seq not in sequences:
        return None
    
    similar_dir = sequences[similar_seq]
    similar_imgs = select_co3d_viewpoints(similar_dir, 1)
    if not similar_imgs:
        return None
    
    img_b = similar_imgs[0]
    img_b_path = os.path.join(similar_dir, img_b)
    
    # Verify all paths exist
    if not all(os.path.exists(p) for p in [img_a_path, img_a_prime_path, img_b_path]):
        return None
    
    # Create condition string - similarity value + category
    condition = f"{similarity_score:.3f}_{category}"
    
    # Create descriptive names
    name_a = f"{category}/{focal_sequence}"
    name_b = f"{category}/{similar_seq}"
    
    return {
        'A': img_a_path,
        'A_prime': img_a_prime_path,
        'B': img_b_path,
        'condition': condition,
        'name_a': name_a,
        'name_b': name_b,
        'similarity_score': similarity_score
    }

def generate_co3d_triplets(base_path: str, similarity_file: str,
                          bg_type: str, ratios: List[float], n_triplets: int):
    """Generate n_triplets for CO3D dataset."""
    print(f"Generating {n_triplets} CO3D triplets ({bg_type} background)...")
    
    # Load similarity data and structure
    similarity_data = load_similarity_data(similarity_file)
    if not similarity_data:
        print(f"❌ Failed to load similarity data from {similarity_file}")
        return []
    
    structure = get_co3d_structure(base_path)
    if not structure:
        print(f"❌ No CO3D structure found in {base_path}")
        return []
    
    # Count categories and sequences
    total_sequences = 0
    for category, sequences in structure.items():
        total_sequences += len(sequences)
    
    print(f"  Found {len(structure)} categories with {total_sequences} total sequences")
    print(f"  Similarity data has {len(similarity_data)} categories")
    
    triplets = []
    bin_types = ['low', 'medium', 'high']
    bin_counts = [int(n_triplets * r) for r in ratios]
    
    # Ensure we generate exactly n_triplets
    total_assigned = sum(bin_counts)
    if total_assigned < n_triplets:
        bin_counts[-1] += n_triplets - total_assigned
    
    for bin_type, target_count in zip(bin_types, bin_counts):
        print(f"  Generating {target_count} {bin_type} similarity triplets...")
        
        attempts = 0
        generated = 0
        max_attempts = target_count * 20
        consecutive_failures = 0
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            triplet = generate_co3d_triplet(structure, similarity_data, bin_type)
            
            if triplet:
                triplet.update({
                    'trial': len(triplets) + 1,
                    'bg': bg_type,
                    'dataset': 'CO3D'
                })
                triplets.append(triplet)
                generated += 1
                consecutive_failures = 0
                
                if generated % 1000 == 0:
                    print(f"    Generated {generated}/{target_count}")
            else:
                consecutive_failures += 1
                if consecutive_failures > 1000:
                    print(f"    ⚠️  Too many consecutive failures, may have issues with data")
                    break
    
    print(f"✅ Generated {len(triplets)} CO3D triplets")
    return triplets

# ──────────────────────────── Shapegen Functions ────────────────────────────

def load_similarity_data(similarity_file: str):
    """Load similarity data from pickle file"""
    try:
        with open(similarity_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Loaded similarity data for {len(data)} items")
        return data
    except Exception as e:
        print(f"❌ Error loading similarity data: {e}")
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

def generate_shapegen_triplet(base_path: str, camera_path: str, structure: Dict, similarity_data: Dict, bin_type: str):
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
        return None
    
    # Select focal object from similarity data keys
    available_shapes = list(structure[extrusion][smoothness].keys())
    similarities = similarity_data[extrusion_key][smoothness_key]
    
    # Find shapes that exist in both structure and similarity data
    valid_shapes = [shape for shape in available_shapes if shape in similarities]
    if len(valid_shapes) < 2:
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
    focal_imgs = select_viewpoints(focal_dir, camera_path, 10)
    if len(focal_imgs) < 2:
        return None
    
    img_a, img_a_prime = random.sample(focal_imgs, 2)
    img_a_path = os.path.join(focal_dir, img_a)
    img_a_prime_path = os.path.join(focal_dir, img_a_prime)
    
    # Get similar object image (B)
    similar_obj, similarity_score = random.choice(bins[bin_index])
    if similar_obj not in structure[extrusion][smoothness]:
        return None
    
    similar_dir = structure[extrusion][smoothness][similar_obj]
    similar_imgs = select_viewpoints(similar_dir, camera_path, 5)
    if not similar_imgs:
        return None
    
    img_b = random.choice(similar_imgs)
    img_b_path = os.path.join(similar_dir, img_b)
    
    # Verify all paths exist
    if not all(os.path.exists(p) for p in [img_a_path, img_a_prime_path, img_b_path]):
        return None
    
    # Create condition string - just the similarity value
    condition = f"{similarity_score:.3f}"
    
    # Create descriptive names with full path context
    name_a = f"extrusion_{extrusion}/smoothness_{smoothness}/{focal_shape}"
    name_b = f"extrusion_{extrusion}/smoothness_{smoothness}/{similar_obj}"
    
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
                              bg_type: str, ratios: List[float], n_triplets: int):
    """Generate n_triplets for Shapegen dataset."""
    print(f"Generating {n_triplets} Shapegen triplets ({bg_type} background)...")
    
    # Load similarity data and structure
    similarity_data = load_similarity_data(similarity_file)
    if not similarity_data:
        print(f"❌ Failed to load similarity data from {similarity_file}")
        return []
    
    structure = get_shapegen_structure(base_path)
    if not structure:
        print(f"❌ No Shapegen structure found in {base_path}")
        return []
    
    triplets = []
    bin_types = ['low', 'medium', 'high']
    bin_counts = [int(n_triplets * r) for r in ratios]
    
    # Ensure we generate exactly n_triplets
    total_assigned = sum(bin_counts)
    if total_assigned < n_triplets:
        bin_counts[-1] += n_triplets - total_assigned
    
    for bin_type, target_count in zip(bin_types, bin_counts):
        print(f"  Generating {target_count} {bin_type} similarity triplets...")
        
        attempts = 0
        generated = 0
        max_attempts = target_count * 20
        consecutive_failures = 0
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            triplet = generate_shapegen_triplet(base_path, camera_path, structure, similarity_data, bin_type)
            
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
                    print(f"    Generated {generated}/{target_count}")
            else:
                consecutive_failures += 1
                if consecutive_failures > 1000:
                    print(f"    ⚠️  Too many consecutive failures, may have issues with data")
                    break
    
    print(f"✅ Generated {len(triplets)} Shapegen triplets")
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

def generate_primigen_triplet(structure: Dict, camera_path: str, condition_type: str, n_level: str = None):
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
    imgs_a = select_viewpoints(dir_a, camera_path, 10)
    if len(imgs_a) < 2:
        return None
    
    imgs_b = select_viewpoints(dir_b, camera_path, 5)
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
                              condition_ratios: List[float], n_ratios: List[float], n_triplets: int):
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
            attempts = 0
            generated = 0
            max_attempts = n_target * 10
            
            while generated < n_target and attempts < max_attempts:
                attempts += 1
                triplet = generate_primigen_triplet(structure, camera_path, condition_type, n_level)
                
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
    
    print(f"✅ Generated {len(triplets)} Primigen triplets")
    return triplets

# ──────────────────────────── Objaverse Functions ────────────────────────────

def get_objaverse_structure(base_path: str):
    """Get Objaverse directory structure: object_id -> path"""
    structure = {}
    if not os.path.isdir(base_path):
        return structure
    
    for obj_dir in sorted(d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))):
        obj_path = os.path.join(base_path, obj_dir)
        # Check if this directory has images subdirectory
        images_path = os.path.join(obj_path, "images")
        if os.path.isdir(images_path):
            structure[obj_dir] = images_path
        elif any(f.endswith('.png') for f in os.listdir(obj_path) if os.path.isfile(os.path.join(obj_path, f))):
            # Images are directly in the object directory
            structure[obj_dir] = obj_path
    
    return structure

def calculate_objaverse_similarity_bins(target_object: str, similarity_data: Dict):
    """Calculate 3 similarity bins for above-mean objects (same as Shapegen)."""
    if target_object not in similarity_data:
        return [], [], [], {}
    
    # Get similarities to all other objects
    object_similarities = []
    other_objects = []
    
    for other_obj, sim_value in similarity_data[target_object].items():
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

def generate_objaverse_triplet(structure: Dict, similarity_data: Dict, bin_type: str):
    """Generate one Objaverse triplet based on similarity bins."""
    # Select focal object from similarity data keys
    available_objects = list(structure.keys())
    
    # Find objects that exist in both structure and similarity data
    valid_objects = [obj for obj in available_objects if obj in similarity_data]
    if len(valid_objects) < 2:
        return None
    
    focal_object = random.choice(valid_objects)
    
    # Get similarity bins
    bin1, bin2, bin3, _ = calculate_objaverse_similarity_bins(
        focal_object, similarity_data
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
    
    # Get focal object images (A and A') using Objaverse-specific selection
    focal_dir = structure[focal_object]
    focal_imgs = select_objaverse_viewpoints(focal_dir, 2)
    if len(focal_imgs) < 2:
        return None
    
    img_a, img_a_prime = focal_imgs[0], focal_imgs[1]
    img_a_path = os.path.join(focal_dir, img_a)
    img_a_prime_path = os.path.join(focal_dir, img_a_prime)
    
    # Get similar object image (B)
    similar_obj, similarity_score = random.choice(bins[bin_index])
    if similar_obj not in structure:
        return None
    
    similar_dir = structure[similar_obj]
    similar_imgs = select_objaverse_viewpoints(similar_dir, 1)
    if not similar_imgs:
        return None
    
    img_b = similar_imgs[0]
    img_b_path = os.path.join(similar_dir, img_b)
    
    # Verify all paths exist
    if not all(os.path.exists(p) for p in [img_a_path, img_a_prime_path, img_b_path]):
        return None
    
    # Create condition string - just the similarity value
    condition = f"{similarity_score:.3f}"
    
    # Create descriptive names
    name_a = focal_object
    name_b = similar_obj
    
    return {
        'A': img_a_path,
        'A_prime': img_a_prime_path,
        'B': img_b_path,
        'condition': condition,
        'name_a': name_a,
        'name_b': name_b,
        'similarity_score': similarity_score
    }

def generate_objaverse_triplets(base_path: str, similarity_file: str, 
                               bg_type: str, ratios: List[float], n_triplets: int):
    """Generate n_triplets for Objaverse dataset."""
    print(f"Generating {n_triplets} Objaverse triplets ({bg_type} background)...")
    
    # Load similarity data and structure
    similarity_data = load_similarity_data(similarity_file)
    if not similarity_data:
        print(f"❌ Failed to load similarity data from {similarity_file}")
        return []
    
    structure = get_objaverse_structure(base_path)
    if not structure:
        print(f"❌ No Objaverse structure found in {base_path}")
        return []
    
    print(f"  Found {len(structure)} Objaverse objects")
    print(f"  Similarity data has {len(similarity_data)} objects")
    
    triplets = []
    bin_types = ['low', 'medium', 'high']
    bin_counts = [int(n_triplets * r) for r in ratios]
    
    # Ensure we generate exactly n_triplets
    total_assigned = sum(bin_counts)
    if total_assigned < n_triplets:
        bin_counts[-1] += n_triplets - total_assigned
    
    for bin_type, target_count in zip(bin_types, bin_counts):
        print(f"  Generating {target_count} {bin_type} similarity triplets...")
        
        attempts = 0
        generated = 0
        max_attempts = target_count * 20
        consecutive_failures = 0
        
        while generated < target_count and attempts < max_attempts:
            attempts += 1
            triplet = generate_objaverse_triplet(structure, similarity_data, bin_type)
            
            if triplet:
                triplet.update({
                    'trial': len(triplets) + 1,
                    'bg': bg_type,
                    'dataset': 'OBJAVERSE'
                })
                triplets.append(triplet)
                generated += 1
                consecutive_failures = 0
                
                if generated % 1000 == 0:
                    print(f"    Generated {generated}/{target_count}")
            else:
                consecutive_failures += 1
                if consecutive_failures > 1000:
                    print(f"    ⚠️  Too many consecutive failures, may have issues with data")
                    break
    
    print(f"✅ Generated {len(triplets)} Objaverse triplets")
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
    
    print(f"✅ Saved to {output_path}")

def generate_hida_dataset_complete(
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
    objaverse_black_path: str,
    objaverse_random_path: str,
    objaverse_white_path: str,
    objaverse_sim_file: str,
    co3d_black_path: str,
    co3d_random_path: str,
    co3d_white_path: str,
    co3d_sim_file: str,
    output_csv_path: str,
    shapegen_ratios: List[float] = [0.20, 0.45, 0.35],  # low, medium, high similarity
    primigen_ratios: List[float] = [0.55, 0.27, 0.18],  # place, warp, config
    objaverse_ratios: List[float] = [0.20, 0.45, 0.35],  # low, medium, high similarity
    co3d_ratios: List[float] = [0.20, 0.45, 0.35],      # low, medium, high similarity
    n_ratios: List[float] = [0.43, 0.42, 0.15],         # n2, n3, n4
    triplets_per_combo: int = 20000,
    include_shapegen: bool = True,
    include_primigen: bool = True,
    include_objaverse: bool = True,
    include_co3d: bool = True
):
    """Generate the complete HIDA dataset with Objaverse and CO3D."""
    print("🚀 Generating Complete HIDA Dataset with Objaverse and CO3D...")
    print(f"Target: {12 * triplets_per_combo:,} total triplets")
    print(f"Shapegen similarity ratios: {shapegen_ratios}")
    print(f"Primigen condition ratios: {primigen_ratios}")
    print(f"Objaverse similarity ratios: {objaverse_ratios}")
    print(f"CO3D similarity ratios: {co3d_ratios}")
    print(f"N-level ratios: {n_ratios}")
    
    all_triplets = []
    
    # Generate triplets for each combination
    try:
        # Shapegen triplets
        if include_shapegen:
            shapegen_black = generate_shapegen_triplets(
                shapegen_black_path, shapegen_black_sim_file, shapegen_camera_path,
                "BLACK", shapegen_ratios, triplets_per_combo
            )
            all_triplets.extend(shapegen_black)

            shapegen_random = generate_shapegen_triplets(
                shapegen_random_path, shapegen_random_sim_file, shapegen_camera_path,
                "RANDOM", shapegen_ratios, triplets_per_combo
            )
            all_triplets.extend(shapegen_random)

            shapegen_white = generate_shapegen_triplets(
                shapegen_white_path, shapegen_white_sim_file, shapegen_camera_path,
                "WHITE", shapegen_ratios, triplets_per_combo
            )
            all_triplets.extend(shapegen_white)

        # Primigen triplets
        if include_primigen:
            primigen_black = generate_primigen_triplets(
                primigen_black_path, primigen_camera_path, "BLACK",
                primigen_ratios, n_ratios, triplets_per_combo
            )
            all_triplets.extend(primigen_black)

            primigen_random = generate_primigen_triplets(
                primigen_random_path, primigen_camera_path, "RANDOM",
                primigen_ratios, n_ratios, triplets_per_combo
            )
            all_triplets.extend(primigen_random)

            primigen_white = generate_primigen_triplets(
                primigen_white_path, primigen_camera_path, "WHITE",
                primigen_ratios, n_ratios, triplets_per_combo
            )
            all_triplets.extend(primigen_white)

        # Objaverse triplets (no camera constraints)
        if include_objaverse:
            objaverse_black = generate_objaverse_triplets(
                objaverse_black_path, objaverse_sim_file,
                "BLACK", objaverse_ratios, triplets_per_combo
            )
            all_triplets.extend(objaverse_black)

            objaverse_random = generate_objaverse_triplets(
                objaverse_random_path, objaverse_sim_file,
                "RANDOM", objaverse_ratios, triplets_per_combo
            )
            all_triplets.extend(objaverse_random)

            objaverse_white = generate_objaverse_triplets(
                objaverse_white_path, objaverse_sim_file,
                "WHITE", objaverse_ratios, triplets_per_combo
            )
            all_triplets.extend(objaverse_white)

        # CO3D triplets (no camera constraints)
        if include_co3d:
            co3d_black = generate_co3d_triplets(
                co3d_black_path, co3d_sim_file,
                "BLACK", co3d_ratios, triplets_per_combo
            )
            all_triplets.extend(co3d_black)

            co3d_random = generate_co3d_triplets(
                co3d_random_path, co3d_sim_file,
                "RANDOM", co3d_ratios, triplets_per_combo
            )
            all_triplets.extend(co3d_random)

            co3d_white = generate_co3d_triplets(
                co3d_white_path, co3d_sim_file,
                "WHITE", co3d_ratios, triplets_per_combo
            )
            all_triplets.extend(co3d_white)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n📊 Generated {len(all_triplets):,} total triplets")
    
    # Shuffle and save
    print("🔀 Shuffling triplets...")
    random.shuffle(all_triplets)
    
    save_to_csv(all_triplets, output_csv_path)
    
    print("✅ Complete HIDA dataset generation done!")

def main():
    parser = argparse.ArgumentParser(description='Generate Complete HIDA Dataset with Objaverse and CO3D')
    
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
    parser.add_argument('--objaverse-black', required=True,
                       help='Path to Objaverse black background images')
    parser.add_argument('--objaverse-random', required=True,
                       help='Path to Objaverse random background images')
    parser.add_argument('--objaverse-white', required=True,
                       help='Path to Objaverse white background images')
    parser.add_argument('--objaverse-sim', required=True,
                       help='Path to Objaverse similarity file (.pkl)')
    parser.add_argument('--co3d-black', required=True,
                       help='Path to CO3D black background images')
    parser.add_argument('--co3d-random', required=True,
                       help='Path to CO3D random background images')
    parser.add_argument('--co3d-white', required=True,
                       help='Path to CO3D white background images')
    parser.add_argument('--co3d-sim', required=True,
                       help='Path to CO3D similarity file (.pkl)')
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    
    # Optional ratios
    parser.add_argument('--shapegen-ratios', nargs=3, type=float, 
                       default=[0.15, 0.45, 0.40],
                       help='Ratios for low/medium/high similarity (default: 0.15 0.45 0.40)')
    parser.add_argument('--primigen-ratios', nargs=3, type=float,
                       default=[0.60, 0.25, 0.15], 
                       help='Ratios for place/warp/config (default: 0.60 0.25 0.15)')
    parser.add_argument('--objaverse-ratios', nargs=3, type=float,
                       default=[0.05, 0.20, 0.75], 
                       help='Ratios for low/medium/high similarity (default: 0.05 0.20 0.75)')
    parser.add_argument('--co3d-ratios', nargs=3, type=float,
                       default=[0.20, 0.45, 0.35], 
                       help='Ratios for low/medium/high similarity (default: 0.20 0.45 0.35)')
    parser.add_argument('--n-ratios', nargs=3, type=float,
                       default=[0.40, 0.40, 0.20],
                       help='Ratios for n2/n3/n4 (default: 0.40 0.40 0.20)')
    parser.add_argument('--triplets-per-combo', type=int, default=20000,
                       help='Number of triplets per combination (default: 20000)')
    parser.add_argument('--seed', type=int, default=42)

    # Dataset inclusion flags
    parser.add_argument('--include-shapegen', action='store_true', default=True,
                       help='Include Shapegen dataset (default: True)')
    parser.add_argument('--no-shapegen', dest='include_shapegen', action='store_false',
                       help='Exclude Shapegen dataset')
    parser.add_argument('--include-primigen', action='store_true', default=True,
                       help='Include Primigen dataset (default: True)')
    parser.add_argument('--no-primigen', dest='include_primigen', action='store_false',
                       help='Exclude Primigen dataset')
    parser.add_argument('--include-objaverse', action='store_true', default=True,
                       help='Include Objaverse dataset (default: True)')
    parser.add_argument('--no-objaverse', dest='include_objaverse', action='store_false',
                       help='Exclude Objaverse dataset')
    parser.add_argument('--include-co3d', action='store_true', default=True,
                       help='Include CO3D dataset (default: True)')
    parser.add_argument('--no-co3d', dest='include_co3d', action='store_false',
                       help='Exclude CO3D dataset')
    
    args = parser.parse_args()
    
    # Validate ratio sums
    for ratios, name in [(args.shapegen_ratios, 'shapegen'), 
                        (args.primigen_ratios, 'primigen'),
                        (args.objaverse_ratios, 'objaverse'),
                        (args.co3d_ratios, 'co3d'),
                        (args.n_ratios, 'n-level')]:
        if abs(sum(ratios) - 1.0) > 0.01:
            print(f"❌ {name} ratios must sum to 1.0, got {sum(ratios)}")
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
        (args.primigen_camera, "Primigen camera"),
        (args.objaverse_black, "Objaverse black"),
        (args.objaverse_random, "Objaverse random"),
        (args.objaverse_white, "Objaverse white"),
        (args.objaverse_sim, "Objaverse similarity file"),
        (args.co3d_black, "CO3D black"),
        (args.co3d_random, "CO3D random"),
        (args.co3d_white, "CO3D white"),
        (args.co3d_sim, "CO3D similarity file")
    ]
    
    for path, name in paths_to_check:
        if not os.path.exists(path):
            print(f"❌ {name} path does not exist: {path}")
            sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    generate_hida_dataset_complete(
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
        args.objaverse_black,
        args.objaverse_random,
        args.objaverse_white,
        args.objaverse_sim,
        args.co3d_black,
        args.co3d_random,
        args.co3d_white,
        args.co3d_sim,
        args.output,
        args.shapegen_ratios,
        args.primigen_ratios,
        args.objaverse_ratios,
        args.co3d_ratios,
        args.n_ratios,
        args.triplets_per_combo,
        args.include_shapegen,
        args.include_primigen,
        args.include_objaverse,
        args.include_co3d
    )

if __name__ == "__main__":
    main()