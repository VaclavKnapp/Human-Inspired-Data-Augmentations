# Human Inspired Data Augmentations (HIDA)

A unified PyTorch training pipeline for HIDA learning

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure your training:**
Edit `config.yaml` to set your dataset paths and training parameters.

3. **Run training:**
```bash
python train.py
```

## Project Structure

```
hida-tune/
├── config.yaml                 # Main configuration file
├── train.py                   # Main training script
├── create_csv.py              # Dataset generation (HIDA only)
├── create_csv_w_Objaverse.py  # Dataset generation (HIDA + Objaverse)
├── datasets/
│   ├── builder.py             # Dataset builder utilities
│   └── hida_dataset.py        # HIDA dataset implementation
├── models/
│   └── dinov2.py             # DINOv2 backbone implementation
├── utils/
│   ├── losses.py             # Loss function implementations
│   └── ...                   # Other utilities
├── configs/                  # Legacy config files (deprecated)
├── logs/                     # Training logs and checkpoints
└── augmented_samples/        # Sample augmented images (for debugging)
```

## Configuration

The main configuration file `config.yaml` contains all training parameters organized into sections:

### Dataset Generation Configuration
Configure which datasets to use and how to generate them:

```yaml
dataset_generation:
  # Options: "hida", "hida_objaverse", "objaverse"
  dataset_type: "hida_objaverse"
  
  # Regenerate dataset for each epoch (optional, for data diversity)
  regenerate_per_epoch: false
  
  paths:
    # Shapegen dataset paths
    shapegen_black: "/path/to/shapegen/black"
    shapegen_random: "/path/to/shapegen/random"
  
  # Number of triplets per dataset+background combination
  triplets_per_combo: 6666
  
  # Similarity/condition ratios for different datasets
  shapegen_ratios: [0.15, 0.45, 0.40]  # [low, medium, high] similarity
  primigen_ratios: [0.60, 0.25, 0.15]  # [place, warp, config] conditions
  objaverse_ratios: [0.05, 0.20, 0.75] # [low, medium, high] similarity
```

### Model Configuration
Configure the backbone and LoRA settings:

```yaml
model:
  backbone:
    _target_: models.dinov2.DINOv2
    checkpoint: "vit_large_patch14_dinov2"  # or vit_base_patch14_dinov2
  
  # LoRA fine-tuning configuration
  use_lora: true
  lora_r: 8
  lora_alpha: 8
  lora_dropout: 0.1
```

### Loss Configuration
Choose and configure your loss function:

```yaml
loss:
  # Options: "triplet", "hinge", "multi_similarity", "oddity"
  type: "oddity"
  
  # Loss-specific parameters
  margin: 0.1        # For triplet/hinge loss
  temperature: 0.1   # For oddity loss
  
  # Multi-similarity loss parameters
  alpha: 2
  beta: 50
  base: 1
```

### Data Augmentation Configuration
Control augmentations applied during training:

```yaml
augmentation:
  enabled: true
  save_every_n: 1000  # Save augmented samples for inspection
  save_dir: "./augmented_samples"
  
  color_jitter:
    brightness: 0.10
    contrast: 0.13
    saturation: 0.1
    hue: 0.05
    probability: 0.5
  
  gaussian_blur:
    kernel_size: 3
    sigma: [0.1, 0.6]
    probability: 0.5
  
```

## Dataset Types

### HIDA Dataset (`dataset_type: "hida"`)
- Uses `create_csv.py`
- Combines Shapegen + Primigen datasets
- 6 combinations: 3 backgrounds × 2 datasets

### HIDA + Objaverse (`dataset_type: "hida_objaverse"`)
- Uses `create_csv_w_Objaverse.py`
- Combines Shapegen + Primigen + Objaverse
- 9 combinations: 3 backgrounds × 3 datasets

### Objaverse Only (`dataset_type: "objaverse"`)
- Uses `create_csv_w_Objaverse.py`
- Only Objaverse dataset
- 3 combinations: 3 backgrounds × 1 dataset

## Architecture Overview

### Training Pipeline

1. **Dataset Generation**: Based on `dataset_type`, generates CSV files with triplet information
2. **Model Creation**: Instantiates DINOv2 backbone with optional LoRA fine-tuning
3. **Data Loading**: Creates train/val/test dataloaders with configured augmentations
4. **Loss Function**: Applies selected loss function (oddity, triplet, etc.)
5. **Training Loop**: Standard PyTorch training with validation and checkpointing

### Key Components

#### ContrastiveModel (`models/`)
Wraps the DINOv2 backbone for contrastive learning:
- Extracts embeddings from triplet inputs (anchor, positive, negative)
- Supports LoRA fine-tuning for efficient training
- Handles different DINOv2 variants (base/large)

#### HIDADataset (`datasets/hida_dataset.py`)
PyTorch Dataset for loading triplet data:
- Loads images from CSV triplet definitions
- Applies configurable augmentations during training
- Supports train/val splits with filtering
- Can save augmented samples for inspection

#### Loss Functions (`utils/losses.py`)
Multiple loss function implementations:
- **TripletLoss**: Classic triplet loss with margin
- **HingeLoss**: Hinge-based triplet loss
- **SingleTripletMultiSimilarityLoss**: Multi-similarity loss
- **Oddity_Loss**: Oddity task loss

#### Data Augmentation (`datasets/hida_dataset.py`)
Configurable augmentation pipeline:
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian blur with configurable sigma
- Motion blur with random angles
- Gaussian noise injection

## Dataset Generation Scripts

### create_csv.py
Generates triplets for HIDA dataset (Shapegen + Primigen):
- **Shapegen**: Uses similarity-based triplet selection with 3 bins
- **Primigen**: Uses condition-based triplets (place/warp/config variations)

### create_csv_w_Objaverse.py
Extended version including Objaverse:
- All functionality of `create_csv.py`
- **Objaverse**: Similarity-based triplets with 360° viewpoint selection

### Generated CSV Format
```csv
Trial,BG,A,A_prime,B,DATASET,CONDITION,NAME_A,NAME_B
1,BLACK,/path/to/anchor.png,/path/to/positive.png,/path/to/negative.png,SHAPEGEN,0.750,shape_001,shape_002
```

## Development

### Adding New Loss Functions
1. Implement loss class in `utils/losses.py`
2. Add case to `get_loss_function()` in `train.py`
3. Update config documentation

### Adding New Augmentations
1. Implement augmentation class in `datasets/hida_dataset.py`
2. Add configuration to `DomainRandomizationTransform.__init__()`
3. Update config schema

### Adding New Backbones
1. Implement backbone class in `models/`
2. Update `ContrastiveModel` to support new backbone
3. Add configuration options


