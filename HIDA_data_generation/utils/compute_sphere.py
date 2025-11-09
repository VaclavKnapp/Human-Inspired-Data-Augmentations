#!/usr/bin/env python3
"""
Generate Fibonacci sphere viewpoints and save them to a pickle file.
These viewpoints provide evenly distributed camera positions around a sphere.
"""

import math
import pickle
import argparse
import os
import random

def generate_fibonacci_sphere(n_points, randomize=False):
    """
    Generate points on a unit sphere using the Fibonacci spiral method.
    
    Args:
        n_points (int): Number of points to generate
        randomize (bool): Add a small random offset to distribute points more evenly
        
    Returns:
        list: List of (x, y, z) tuples representing points on a unit sphere
    """
    points = []
    
    # Golden ratio φ for optimal spacing
    phi = (1 + math.sqrt(5)) / 2
    
    for i in range(n_points):
        # Use a safer formula to ensure t stays within [-1+ε, 1-ε]
        t = -1 + (2 * i) / (max(1, n_points - 1))
        if randomize:
            # Apply randomization but ensure we stay within valid bounds
            rand_offset = random.uniform(-0.00005, 0.00005)
            t = max(-0.9999, min(0.9999, t + rand_offset))
        
        # Convert to spherical coordinates
        z = t
        
        # Calculate radius at this latitude (from Pythagorean identity)
        radius = math.sqrt(1 - z*z)
        
        # Calculate the golden angle (in radians)
        theta = 2 * math.pi * i / phi
        
        # Convert to Cartesian coordinates
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        
        points.append((x, y, z))
    
    return points

def save_viewpoints(points, filename):
    """Save viewpoints to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(points, f)
    print(f"Saved {len(points)} viewpoints to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate Fibonacci sphere viewpoints')
    parser.add_argument('n_points', type=int, help='Number of viewpoints to generate')
    parser.add_argument('--output', '-o', default='fibonacci_viewpoints.pkl', 
                        help='Output pickle file path')
    parser.add_argument('--randomize', '-r', action='store_true', 
                        help='Add small random offsets for better distribution')
    
    args = parser.parse_args()
    
    # Generate viewpoints
    viewpoints = generate_fibonacci_sphere(args.n_points, args.randomize)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    
    # Save to pickle file
    save_viewpoints(viewpoints, args.output)

if __name__ == "__main__":
    main()