#!/usr/bin/env python3
"""
Apply backgrounds to transparent rendered images.

This script takes transparent renders and applies backgrounds to them
in the same way as the original Blender rendering script did.
"""

import os
import sys
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply backgrounds to transparent rendered images."
    )
    parser.add_argument(
        "render_dir", 
        help="Directory containing the transparent rendered images"
    )
    
    # Create mutually exclusive group for background options
    bg_group = parser.add_mutually_exclusive_group(required=True)
    bg_group.add_argument(
        "bg_dir", 
        nargs='?',
        help="Directory containing background images"
    )
    bg_group.add_argument(
        "--white", 
        action="store_true",
        help="Use a solid white background"
    )
    bg_group.add_argument(
        "--black", 
        action="store_true",
        help="Use a solid black background"
    )
    
    parser.add_argument(
        "output_dir", 
        help="Directory where composited images will be saved"
    )
    parser.add_argument(
        "--random-bg", 
        action="store_true", 
        help="Use a random background for each image (default: use same bg for all images in a folder)"
    )
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=518, 
        help="Target image size (default: 518px)"
    )
    parser.add_argument(
        "--zoom-range", 
        type=float, 
        nargs=2, 
        default=[0.5, 1.5], 
        help="Random zoom range [min, max] (default: 0.7 1.3)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print detailed processing information"
    )
    return parser.parse_args()

def get_background_files(bg_dir):
    """Get list of background image files with png extension."""
    background_files = []
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        background_files.extend(list(Path(bg_dir).glob(f"*{ext}")))
    
    if not background_files:
        print(f"Error: No image files found in {bg_dir}")
        sys.exit(1)
    
    return background_files

def create_solid_background(color, target_size):
    """Create a solid color background image."""
    if color == 'white':
        bg_color = (255, 255, 255, 255)
    elif color == 'black':
        bg_color = (0, 0, 0, 255)
    else:
        raise ValueError(f"Unsupported color: {color}")
    
    # Create a new RGBA image with the specified color
    bg_img = Image.new('RGBA', (target_size, target_size), bg_color)
    return bg_img

def crop_square_from_center(image):
    """Crop a square from the center of the image."""
    width, height = image.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))

def process_background(bg_path, target_size):
    """Process background image: load, crop to square, resize."""
    try:
        # Open the background image
        bg_img = Image.open(bg_path)
        
        # Crop a square from the center
        bg_img = crop_square_from_center(bg_img)
        
        # Resize to target size
        bg_img = bg_img.resize((target_size, target_size), Image.LANCZOS)
        
        # Convert to RGBA if it's not already
        if bg_img.mode != 'RGBA':
            bg_img = bg_img.convert('RGBA')
        
        return bg_img
    except Exception as e:
        print(f"Error processing background {bg_path}: {e}")
        return None

def apply_random_zoom(render_img, zoom_range, target_size):
    """Apply random zoom to render image while maintaining transparency."""
    zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
    
    # Calculate new size based on zoom factor
    original_size = render_img.size
    new_size = (int(original_size[0] * zoom_factor), int(original_size[1] * zoom_factor))
    
    # Resize the image
    zoomed_img = render_img.resize(new_size, Image.LANCZOS)
    
    # If the zoomed image is larger than target, scale it down to fit
    if new_size[0] > target_size or new_size[1] > target_size:
        # Calculate scale factor to fit within target size
        scale_factor = min(target_size / new_size[0], target_size / new_size[1])
        final_size = (int(new_size[0] * scale_factor), int(new_size[1] * scale_factor))
        zoomed_img = zoomed_img.resize(final_size, Image.LANCZOS)
        new_size = final_size
    
    # Create a new image with target size and transparent background
    result = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
    
    # Calculate position to center the zoomed image
    x_offset = (target_size - new_size[0]) // 2
    y_offset = (target_size - new_size[1]) // 2
    
    # Paste the zoomed image onto the result
    result.paste(zoomed_img, (x_offset, y_offset), zoomed_img)
    
    return result

def composite_images(render_img, bg_img):
    """Composite transparent render over background."""
    # Ensure both images are RGBA
    if render_img.mode != 'RGBA':
        render_img = render_img.convert('RGBA')
    if bg_img.mode != 'RGBA':
        bg_img = bg_img.convert('RGBA')
    
    # Create a new image with the same size as render
    result = Image.new('RGBA', render_img.size)
    
    # Paste background first
    result.paste(bg_img, (0, 0))
    
    # Composite render on top using its alpha channel
    result.alpha_composite(render_img)
    
    return result

def count_total_files(render_dir):
    """Count total number of PNG files to process."""
    total_files = 0
    for root, _, _ in os.walk(render_dir):
        png_files = list(Path(root).glob('*.png'))
        total_files += len(png_files)
    return total_files

def process_render_folder(render_folder, bg_files, output_dir, args, pbar=None):
    """Process all rendered images in a folder."""
    render_files = list(Path(render_folder).glob('*.png'))
    
    if not render_files:
        if args.verbose:
            print(f"No PNG files found in {render_folder}")
        return
    
    # Create output directory with the same structure
    rel_path = os.path.relpath(render_folder, args.render_dir)
    output_folder = os.path.join(args.output_dir, rel_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Determine background type and prepare accordingly
    if args.white or args.black:
        # Use solid color background
        color = 'white' if args.white else 'black'
        processed_bg = create_solid_background(color, args.target_size)
        if args.verbose:
            print(f"Using solid {color} background")
    else:
        # Use image backgrounds
        chosen_bg = random.choice(bg_files)
        if args.verbose:
            print(f"Selected background: {chosen_bg}")
            
        processed_bg = process_background(chosen_bg, args.target_size)
        if processed_bg is None:
            print(f"Error: Could not process background {chosen_bg}")
            if pbar:
                pbar.update(len(render_files))  # Update progress even if we skip
            return
    
    # Process each render with individual progress updates
    for render_file in render_files:
        try:
            # If using random backgrounds and image files, select a different one for each image
            if args.random_bg and not (args.white or args.black):
                chosen_bg = random.choice(bg_files)
                processed_bg = process_background(chosen_bg, args.target_size)
                if processed_bg is None:
                    if pbar:
                        pbar.update(1)
                    continue
            
            # Load render image
            render_img = Image.open(render_file)
            
            # Apply random zoom to render image
            render_img = apply_random_zoom(render_img, args.zoom_range, args.target_size)
            
            # Composite images
            result = composite_images(render_img, processed_bg)
            
            # Save result
            output_file = os.path.join(output_folder, render_file.name)
            result.save(output_file)
            
            if args.verbose:
                print(f"Processed: {render_file.name} -> {output_file}")
                
        except Exception as e:
            print(f"Error processing {render_file}: {e}")
        
        # Update progress bar
        if pbar:
            pbar.update(1)

def main():
    """Main function."""
    args = parse_arguments()
    
    # Check if render directory exists
    if not os.path.isdir(args.render_dir):
        print(f"Error: Render directory '{args.render_dir}' does not exist.")
        sys.exit(1)
    
    # Check background directory only if using image backgrounds
    if args.bg_dir:
        if not os.path.isdir(args.bg_dir):
            print(f"Error: Background directory '{args.bg_dir}' does not exist.")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get background files only if using image backgrounds
    bg_files = None
    if args.bg_dir:
        print("Loading background images...")
        bg_files = get_background_files(args.bg_dir)
        if args.verbose:
            print(f"Found {len(bg_files)} background images in {args.bg_dir}")
    elif args.white:
        print("Using solid white backgrounds...")
    elif args.black:
        print("Using solid black backgrounds...")
    
    # Count total files to process
    print("Counting files to process...")
    total_files = count_total_files(args.render_dir)
    
    if total_files == 0:
        print("No PNG files found to process.")
        return
    
    print(f"Found {total_files} PNG files to process.")
    
    # Create progress bar
    with tqdm(total=total_files, desc="Processing images", unit="file") as pbar:
        # Walk through render directory and process all folders with PNG files
        for root, _, _ in os.walk(args.render_dir):
            if args.verbose:
                tqdm.write(f"Processing folder: {root}")
            process_render_folder(root, bg_files, args.output_dir, args, pbar)
    
    print(f"Background application complete! All files saved to {args.output_dir}")

if __name__ == "__main__":
    main()