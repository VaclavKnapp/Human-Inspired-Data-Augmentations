import os
import random
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple

def oddity_detection_with_precomputed_pairs(pairs_A: List[Tuple[str, str]], 
                                           features_dict: Dict[str, np.ndarray],
                                           image_B_path: str) -> bool:
    """
    Run oddity detection using precomputed pairs and features.
    Returns True if model correctly identifies the odd image, False otherwise.
    """
    # Select a random valid pair from object A
    img_A1, img_A2 = random.choice(pairs_A)
    
    # Get features
    if img_A1 not in features_dict or img_A2 not in features_dict or image_B_path not in features_dict:
        return False
    
    features_A1 = features_dict[img_A1]
    features_A2 = features_dict[img_A2]
    features_B = features_dict[image_B_path]
    
    # Create feature matrix
    features = np.vstack([features_A1, features_A2, features_B])
    
    # Track which is the odd one
    b_index = 2  # B is at index 2
    
    # Shuffle order
    indices = list(range(3))
    random.shuffle(indices)
    features = features[indices]
    b_index = indices.index(b_index)
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(features)
    
    # Calculate average correlation with other images for each image
    off_diag_corr_0 = (similarity_matrix[0, 1] + similarity_matrix[0, 2]) / 2
    off_diag_corr_1 = (similarity_matrix[1, 0] + similarity_matrix[1, 2]) / 2
    off_diag_corr_2 = (similarity_matrix[2, 0] + similarity_matrix[2, 1]) / 2
    
    correlations = [off_diag_corr_0, off_diag_corr_1, off_diag_corr_2]
    b_index_guess = np.argmin(correlations)
    
    return b_index_guess == b_index

def get_all_images_for_object(valid_pairs: Dict, key: Tuple[str, str, str]) -> List[str]:
    """Get all unique images for an object from its valid pairs"""
    if key not in valid_pairs:
        return []
    
    unique_images = set()
    for img1, img2 in valid_pairs[key]:
        unique_images.add(img1)
        unique_images.add(img2)
    
    return list(unique_images)

def measure_object_pair_similarity_with_pairs(valid_pairs: Dict,
                                             features_dict: Dict[str, np.ndarray],
                                             key_A: Tuple[str, str, str],
                                             key_B: Tuple[str, str, str],
                                             num_trials: int = 50) -> float:
    """
    Measure similarity between two objects using precomputed pairs and features.
    Returns similarity score (1 - oddity_accuracy).
    """
    pairs_A = valid_pairs.get(key_A, [])
    pairs_B = valid_pairs.get(key_B, [])
    
    if len(pairs_A) == 0 or len(pairs_B) == 0:
        print(f"Warning: No valid pairs for one of the objects")
        return 0.5
    
    # Get all images for object B (for selecting the odd one out)
    images_B = get_all_images_for_object(valid_pairs, key_B)
    images_A = get_all_images_for_object(valid_pairs, key_A)
    
    if len(images_B) == 0 or len(images_A) == 0:
        return 0.5
    
    accuracies = []
    
    for trial in range(num_trials):
        try:
            # Run oddity detection (2 from A, 1 from B)
            image_B = random.choice(images_B)
            is_correct_A_to_B = oddity_detection_with_precomputed_pairs(
                pairs_A, features_dict, image_B
            )
            
            # Run oddity detection (2 from B, 1 from A)
            image_A = random.choice(images_A)
            is_correct_B_to_A = oddity_detection_with_precomputed_pairs(
                pairs_B, features_dict, image_A
            )
            
            # Average the two directions
            avg_accuracy = (int(is_correct_A_to_B) + int(is_correct_B_to_A)) / 2
            accuracies.append(avg_accuracy)
            
        except Exception as e:
            print(f"Warning: Oddity detection failed in trial {trial}: {e}")
            continue
    
    if len(accuracies) == 0:
        print("Error: All trials failed")
        return 0.5
    
    # Convert oddity accuracy to similarity
    avg_oddity_accuracy = np.mean(accuracies)
    similarity = 1 - avg_oddity_accuracy
    
    return similarity

def compute_similarities_with_precomputed_pairs(valid_pairs_file: str,
                                              features_file: str,
                                              num_trials: int = 50) -> Dict:
    """Compute object similarities using precomputed pairs and features"""
    
    # Load precomputed pairs
    print("Loading precomputed pairs...")
    with open(valid_pairs_file, 'rb') as f:
        valid_pairs = pickle.load(f)
    
    # Load features
    print("Loading extracted features...")
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    
    print(f"Loaded {len(valid_pairs)} objects with valid pairs")
    print(f"Loaded features for {len(features_dict)} images")
    
    # Organize by condition
    conditions = {}
    for key in valid_pairs.keys():
        extrusion, smoothness, shape_id = key
        condition = (extrusion, smoothness)
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(shape_id)
    
    # Count total pairs
    total_pairs = 0
    for condition, shapes in conditions.items():
        n_shapes = len(shapes)
        pairs_in_condition = n_shapes * (n_shapes - 1) // 2
        total_pairs += pairs_in_condition
    
    print(f"\nComputing similarities for {total_pairs} object pairs")
    print(f"Trials per pair: {num_trials}")
    
    # Results structure
    results = {}
    
    # Progress bar
    pbar = tqdm(total=total_pairs, desc="Computing similarities")
    
    for (extrusion, smoothness), shapes in sorted(conditions.items()):
        if f'extrusion_{extrusion}' not in results:
            results[f'extrusion_{extrusion}'] = {}
        
        condition_name = f"extrusion_{extrusion}/smoothness_{smoothness}"
        shape_similarities = {}
        
        # Initialize
        for shape_id in shapes:
            shape_similarities[f'shape_{shape_id}'] = {}
        
        # Compute pairwise similarities
        for i, shape_A in enumerate(shapes):
            for j, shape_B in enumerate(shapes):
                if i == j:
                    continue
                elif i > j:
                    continue
                
                key_A = (extrusion, smoothness, shape_A)
                key_B = (extrusion, smoothness, shape_B)
                
                pbar.set_description(f"{condition_name}: shape_{shape_A} vs shape_{shape_B}")
                
                # Measure similarity
                similarity = measure_object_pair_similarity_with_pairs(
                    valid_pairs, features_dict, key_A, key_B, num_trials
                )
                
                # Store in both directions
                shape_similarities[f'shape_{shape_A}'][f'shape_{shape_B}'] = similarity
                shape_similarities[f'shape_{shape_B}'][f'shape_{shape_A}'] = similarity
                
                pbar.update(1)
                pbar.set_postfix({"Similarity": f"{similarity:.3f}"})
        
        results[f'extrusion_{extrusion}'][f'smoothness_{smoothness}'] = shape_similarities
    
    pbar.close()
    
    return results

# Visualization functions (same as before)
def create_similarity_matrices(results, output_path):
    """Create and save similarity matrices for each extrusion/smoothness combination."""
    plots_dir = os.path.join(output_path, 'similarity_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Creating similarity matrices...")
    
    for extrusion_key in sorted(results.keys()):
        for smoothness_key in sorted(results[extrusion_key].keys()):
            condition_dir = os.path.join(plots_dir, extrusion_key, smoothness_key)
            os.makedirs(condition_dir, exist_ok=True)
            
            similarities = results[extrusion_key][smoothness_key]
            
            if not similarities:
                continue
            
            shape_ids = sorted(list(similarities.keys()))
            n_shapes = len(shape_ids)
            
            if n_shapes == 0:
                continue
            
            # Create similarity matrix
            matrix = np.ones((n_shapes, n_shapes))
            
            for i, shape_i in enumerate(shape_ids):
                for j, shape_j in enumerate(shape_ids):
                    if i != j:
                        matrix[i, j] = similarities[shape_i].get(shape_j, 0.0)
            
            # Plot matrix
            plt.figure(figsize=(12, 10))
            mask = np.zeros_like(matrix, dtype=bool)
            np.fill_diagonal(mask, True)
            
            sns.heatmap(matrix, mask=mask, xticklabels=shape_ids, yticklabels=shape_ids, 
                       annot=True, fmt='.3f', cmap='viridis', vmin=0, vmax=1,
                       cbar_kws={'label': 'Similarity Score'})
            plt.title(f'Object Similarity Matrix - {extrusion_key}/{smoothness_key}')
            plt.xlabel('Shape ID')
            plt.ylabel('Shape ID')
            plt.tight_layout()
            
            filename = f'similarity_matrix_{extrusion_key}_{smoothness_key}.png'
            filepath = os.path.join(condition_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Created matrix: {extrusion_key}/{smoothness_key}")

def create_similarity_histograms(results, output_path):
    """Create histograms of similarity distributions."""
    plots_dir = os.path.join(output_path, 'similarity_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Creating similarity histograms...")
    
    all_similarities = []
    
    for extrusion_key in sorted(results.keys()):
        for smoothness_key in sorted(results[extrusion_key].keys()):
            condition_dir = os.path.join(plots_dir, extrusion_key, smoothness_key)
            os.makedirs(condition_dir, exist_ok=True)
            
            similarities = results[extrusion_key][smoothness_key]
            
            sim_values = []
            for shape_i in similarities:
                for shape_j in similarities[shape_i]:
                    sim_values.append(similarities[shape_i][shape_j])
            
            if len(sim_values) == 0:
                continue
            
            all_similarities.extend(sim_values)
            
            # Individual histogram
            plt.figure(figsize=(8, 6))
            counts, bins, patches = plt.hist(sim_values, bins=20, alpha=0.7, 
                                           edgecolor='black', density=False)
            plt.xlabel('Similarity Score')
            plt.ylabel('Count')
            plt.title(f'Similarity Distribution - {extrusion_key}/{smoothness_key}')
            plt.grid(True, alpha=0.3)
            
            mean_sim = np.mean(sim_values)
            std_sim = np.std(sim_values)
            plt.axvline(mean_sim, color='red', linestyle='--', 
                       label=f'Mean: {mean_sim:.3f}±{std_sim:.3f}')
            plt.legend()
            
            max_count = np.max(counts)
            total_pairs = len(sim_values)
            plt.text(0.02, 0.98, f'Total pairs: {total_pairs}\nMax count: {int(max_count)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            filename = f'similarity_hist_{extrusion_key}_{smoothness_key}.png'
            filepath = os.path.join(condition_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Created histogram: {extrusion_key}/{smoothness_key}")
    
    # Overall histogram
    if len(all_similarities) > 0:
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(all_similarities, bins=30, alpha=0.7, 
                                        edgecolor='black', density=False)
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plt.title('Overall Similarity Distribution (All Conditions)')
        plt.grid(True, alpha=0.3)
        
        mean_sim = np.mean(all_similarities)
        std_sim = np.std(all_similarities)
        plt.axvline(mean_sim, color='red', linestyle='--', 
                   label=f'Overall Mean: {mean_sim:.3f}±{std_sim:.3f}')
        plt.legend()
        
        max_count = np.max(counts)
        total_pairs = len(all_similarities)
        plt.text(0.02, 0.98, f'Total pairs: {total_pairs}\nMax count: {int(max_count)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        overall_filepath = os.path.join(plots_dir, 'similarity_hist_overall.png')
        plt.savefig(overall_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSimilarity Statistics:")
        print(f"Total pairs analyzed: {total_pairs}")
        print(f"Max count in any bin: {int(max_count)}")
        print(f"Overall Mean: {mean_sim:.3f}")
        print(f"Overall Std: {std_sim:.3f}")
        print(f"Min: {np.min(all_similarities):.3f}")
        print(f"Max: {np.max(all_similarities):.3f}")
        print(f"Median: {np.median(all_similarities):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure similarities using precomputed pairs')
    parser.add_argument('--valid_pairs_file', type=str, required=True,
                        help='Path to valid_pairs.pkl file')
    parser.add_argument('--features_file', type=str, required=True,
                        help='Path to all_features.pkl file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for results and visualizations')
    parser.add_argument('--num_trials', type=int, default=100,
                        help='Number of oddity trials per object pair (default: 25)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Compute similarities
    print(f"Computing similarities with {args.num_trials} trials per pair...")
    similarities = compute_similarities_with_precomputed_pairs(
        args.valid_pairs_file,
        args.features_file,
        args.num_trials
    )
    
    # Save results
    results_file = os.path.join(args.output_path, 'object_similarities_precomputed.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(similarities, f)
    print(f"\nResults saved to {results_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_similarity_matrices(similarities, args.output_path)
    create_similarity_histograms(similarities, args.output_path)
    
    print(f"\nAll done! Check the '{args.output_path}/similarity_plots' directory for visualizations.")