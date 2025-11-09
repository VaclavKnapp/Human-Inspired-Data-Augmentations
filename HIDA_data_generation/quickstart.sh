#!/bin/bash
# HIDA Data Generation - Quick Start Script
# This script provides common configurations for quick testing

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}HIDA Data Generation Quick Start${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

# Parse command line arguments
PRESET="full"
OUTPUT_DIR="hida_output"

show_help() {
    cat << EOF
Usage: ./quickstart.sh [OPTIONS]

Quick start presets for HIDA data generation:

  --preset <name>     Choose a preset configuration:
                      - test      : Minimal dataset for testing (fast)
                      - small     : Small dataset for experimentation
                      - medium    : Medium-sized dataset (default)
                      - full      : Full dataset (slow, production)

  --output <dir>      Output directory (default: hida_output)

  --help              Show this help message

Examples:
  ./quickstart.sh --preset test
  ./quickstart.sh --preset full --output my_dataset

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Define presets
case $PRESET in
    test)
        print_info "Using TEST preset (minimal, fast)"
        ARGS=(
            --output "$OUTPUT_DIR"
            --n-shapes 10
            --smoothness-levels 2
            --extrusions 5
            --n-primitives 2
            --n-configs 2
            --n-warps 2
            --n-placements 3
            --n-viewpoints 10
            --n-lights 4
            --n-textures 10
        )
        ;;
    small)
        print_info "Using SMALL preset (experimental)"
        ARGS=(
            --output "$OUTPUT_DIR"
            --n-shapes 50
            --smoothness-levels 2
            --extrusions 9
            --n-primitives 2 3
            --n-configs 3
            --n-warps 3
            --n-placements 5
            --n-viewpoints 20
            --n-lights 8
            --n-textures 50
        )
        ;;
    medium)
        print_info "Using MEDIUM preset (balanced)"
        ARGS=(
            --output "$OUTPUT_DIR"
            --n-shapes 100
            --smoothness-levels 2
            --extrusions 9
            --n-primitives 2 3 4
            --n-configs 4
            --n-warps 4
            --n-placements 10
            --n-viewpoints 50
            --n-lights 8
            --n-textures 100
        )
        ;;
    full)
        print_info "Using FULL preset (production, slow)"
        ARGS=(
            --output "$OUTPUT_DIR"
            --n-shapes 100
            --smoothness-levels 2
            --extrusions 9
            --n-primitives 2 3 4
            --n-configs 6
            --n-warps 6
            --n-placements 15
            --n-viewpoints 50
            --n-lights 8
            --n-textures 100
        )
        ;;
    *)
        print_error "Unknown preset: $PRESET"
        print_info "Available presets: test, small, medium, full"
        exit 1
        ;;
esac

print_info "Output directory: $OUTPUT_DIR"
echo ""

# Confirm before proceeding
read -p "Proceed with generation? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Aborted by user"
    exit 0
fi

# Run the pipeline
print_info "Starting HIDA data generation pipeline..."
echo ""

python3 "$SCRIPT_DIR/hida_generate.py" "${ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    print_success "Pipeline completed successfully!"
    print_info "Output saved to: $OUTPUT_DIR"
    echo ""
    print_info "Next steps:"
    echo "  1. Check the generated data in: $OUTPUT_DIR"
    echo "  2. Review pipeline_config.json for reproducibility"
    echo "  3. Use the renders for training with main HIDA pipeline"
else
    echo ""
    print_error "Pipeline failed with exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi
