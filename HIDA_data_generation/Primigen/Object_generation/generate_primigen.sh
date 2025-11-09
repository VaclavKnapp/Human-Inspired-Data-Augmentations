#!/bin/bash

# Base directories and settings
SHAPE_CODE_PATH="/home/vaclav_knapp/zeroverse/primishapegen.py"  
OUTPUT_BASE_DIR="/datasets/hida/current/primigen/blender"
TEXTURE_FOLDER="/home/vaclav_knapp/texture_exports"
TRANSLATION_CONTROL=0.8
NUM_SHAPES=1


# Create a function to generate random shape counts
generate_random_shape_counts() {
  local n_objects=$1
  local CYLINDER_COUNT
  local CUBE_COUNT
  local ELLIPSOID_COUNT
  local GEN_COUNT
  local SHAPE_COUNTS
  
  while true; do
    # Always include exactly 1 Generated shape
    local GEN_COUNT=0
    
    # Generate primitive shape counts (n_objects stays the same, we add 1 generated shape on top)
    CYLINDER_COUNT=$(( RANDOM % (n_objects + 1) ))
    CUBE_COUNT=$(( RANDOM % (n_objects + 1) ))
    ELLIPSOID_COUNT=$(( n_objects - CYLINDER_COUNT - CUBE_COUNT ))
    
    if [ "$ELLIPSOID_COUNT" -ge 0 ]; then
      SHAPE_COUNTS="cylinder:$CYLINDER_COUNT,cube:$CUBE_COUNT,ellipsoid:$ELLIPSOID_COUNT,shapegen:$GEN_COUNT"
    else
      # Invalid distribution, try again
      continue
    fi
    
    # Check if this configuration is already in our used list
    local is_duplicate=0
    for used_config in "${USED_CONFIGS[@]}"; do
      if [ "$used_config" == "$SHAPE_COUNTS" ]; then
        is_duplicate=1
        break
      fi
    done
    
    # If not a duplicate, accept this configuration
    if [ $is_duplicate -eq 0 ]; then
      break
    fi
    # Otherwise keep generating
  done
  
  # Add to our used configs list
  USED_CONFIGS+=("$SHAPE_COUNTS")
  echo "$SHAPE_COUNTS"
}

# Create base output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Loop through different numbers of shapes (2-4)
for n_objects in {2..4}; do
  echo "Processing n_objects = $n_objects"
  N_DIR="${OUTPUT_BASE_DIR}/n${n_objects}"
  mkdir -p "$N_DIR"
  
  # Initialize array to track used configurations for this n_objects
  declare -a USED_CONFIGS=()
  
  # For each n_objects, create 5 different configurations
  for config_idx in {1..6}; do
    # Generate a unique shape count configuration
    SHAPE_COUNTS=$(generate_random_shape_counts $n_objects)
    CONFIG_DIR="${N_DIR}/config_${config_idx}_$(echo $SHAPE_COUNTS | sed 's/,/_/g')"
    mkdir -p "$CONFIG_DIR"
    echo "  Configuration ${config_idx}: $SHAPE_COUNTS"
    
    # Generate texture seed for this configuration
    TEXTURE_SEED=$((RANDOM))
    
    # Generate seeds for GeneratedShape parameters
    
    # For each configuration, create 3 different warping variations
    for warp_idx in {1..6}; do
      WARP_DIR="${CONFIG_DIR}/warp_${warp_idx}"
      mkdir -p "$WARP_DIR"
      
      # Generate a unique size seed for this warping
      SIZE_SEED=$((RANDOM))
      # Generate a unique rotation seed
      N_SUBDIVISIONS=$((RANDOM % 6))
      N_EXTRUSIONS_MIN=4
      N_EXTRUSIONS_MAX=6
      echo "    Warp ${warp_idx}: Size Seed ${SIZE_SEED}, Rotation Seed ${ROTATION_SEED}"
      
      # For each warping, create 5 different placement variations
      for place_idx in {1..15}; do
        PLACE_DIR="${WARP_DIR}/place_${place_idx}"
        mkdir -p "$PLACE_DIR"
        
        # Use a different main seed for each placement
        MAIN_SEED=$((RANDOM))
        ROTATION_SEED=$((RANDOM))
        # Generate a unique translation seed
        TRANSLATION_SEED=$((RANDOM))
        
        echo "      Place ${place_idx}: Main Seed ${MAIN_SEED}, Translation Seed ${TRANSLATION_SEED}"
        
        # Create the shape with specific parameters
        python "$SHAPE_CODE_PATH" \
          --num_shapes $NUM_SHAPES \
          --output_dir "$PLACE_DIR" \
          --shape_counts "$SHAPE_COUNTS" \
          --seed $MAIN_SEED \
          --size_seed $SIZE_SEED \
          --rotation_seed $ROTATION_SEED \
          --translation_seed $TRANSLATION_SEED \
          --translation_control $TRANSLATION_CONTROL \
          --texture_folder "$TEXTURE_FOLDER" \
          --texture_seed $TEXTURE_SEED \
          --shapegen_min_extrusions "$N_EXTRUSIONS_MIN" \
          --shapegen_max_extrusions "$N_EXTRUSIONS_MAX" \
          --shapegen_subdivisions "$N_SUBDIVISIONS"
        
        # Rename the output file to a more descriptive name
        mv "${PLACE_DIR}/object_000.blend" "${PLACE_DIR}/n${n_objects}_c${config_idx}_w${warp_idx}_p${place_idx}.blend"
        # Also rename the pickle file
        if [ -f "${PLACE_DIR}/object_000.pkl" ]; then
          mv "${PLACE_DIR}/object_000.pkl" "${PLACE_DIR}/n${n_objects}_c${config_idx}_w${warp_idx}_p${place_idx}.pkl"
        fi
        
        # Create a metadata file with the parameters used
        echo "Shape Count: $SHAPE_COUNTS" > "${PLACE_DIR}/metadata.txt"
        echo "Main Seed: $MAIN_SEED" >> "${PLACE_DIR}/metadata.txt"
        echo "Size Seed: $SIZE_SEED" >> "${PLACE_DIR}/metadata.txt"
        echo "Rotation Seed: $ROTATION_SEED" >> "${PLACE_DIR}/metadata.txt"
        echo "Translation Seed: $TRANSLATION_SEED" >> "${PLACE_DIR}/metadata.txt"
        echo "Texture Seed: $TEXTURE_SEED" >> "${PLACE_DIR}/metadata.txt"
        echo "N Extrusions Range: ${N_EXTRUSIONS_MIN}-${N_EXTRUSIONS_MAX}" >> "${PLACE_DIR}/metadata.txt" 
        echo "Smoothness: ${N_SUBDIVISIONS}" >> "${PLACE_DIR}/metadata.txt"
      done
    done
  done
  
  # Clean up the USED_CONFIGS array before the next n_objects
  unset USED_CONFIGS
done
