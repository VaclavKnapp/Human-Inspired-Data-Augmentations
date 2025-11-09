#!/bin/bash
# Minimal example - Generate a small dataset for testing

set -e

echo "======================================"
echo "HIDA Minimal Example"
echo "======================================"
echo ""
echo "This script generates a minimal dataset for testing:"
echo "- 10 Shapegen objects (1 smoothness level)"
echo "- 18 Primigen objects (n=2 only, minimal hierarchy)"
echo "- 10 viewpoints per object"
echo "- 40 procedural textures"
echo ""
echo "Estimated time: ~15 minutes"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/minimal_output"

echo "Output directory: $OUTPUT_DIR"
echo ""

read -p "Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate viewpoints
echo ""
echo "Step 1/5: Generating viewpoints..."
python3 "$SCRIPT_DIR/utils/compute_sphere.py" 10 \
    --output "$OUTPUT_DIR/viewpoints.pkl"

# Step 2: Generate textures
echo ""
echo "Step 2/5: Generating textures..."
BLENDER=$(which blender || echo "/home/vaclav_knapp/blender-3.6.19-linux-x64/blender")
mkdir -p "$OUTPUT_DIR/textures"
cd "$OUTPUT_DIR/textures"
$BLENDER --background --python "$SCRIPT_DIR/utils/create_textures.py" -- 10
cd "$SCRIPT_DIR"

# Step 3: Generate Shapegen objects
echo ""
echo "Step 3/5: Generating Shapegen objects..."
mkdir -p "$OUTPUT_DIR/shapegen_objects"
$BLENDER --background \
    --python "$SCRIPT_DIR/Shapegen/Object_generation/generate_stim_final_w_pkl.py" -- \
    --output "$OUTPUT_DIR/shapegen_objects" \
    --n_shapes 10 \
    --n_extrusions 5 \
    --smoothness_levels 1 \
    --textures "$OUTPUT_DIR/textures"

# Step 4: Render Shapegen
echo ""
echo "Step 4/5: Rendering Shapegen objects..."
bash "$SCRIPT_DIR/Shapegen/Rendering/render_simplified.sh" \
    "$OUTPUT_DIR/shapegen_objects" \
    "$OUTPUT_DIR/shapegen_renders" \
    --4_lights \
    --random_dim \
    --sphere=true \
    --sphere_file="$OUTPUT_DIR/viewpoints.pkl" \
    --scale_camera=true

echo ""
echo "======================================"
echo "✓ Minimal example completed!"
echo "======================================"
echo ""
echo "Output structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── viewpoints.pkl"
echo "    ├── textures/"
echo "    ├── shapegen_objects/"
echo "    └── shapegen_renders/"
echo ""
echo "Next steps:"
echo "  1. Inspect the rendered images"
echo "  2. Use these images for training/testing"
echo "  3. Scale up with quickstart.sh or hida_generate.py"
