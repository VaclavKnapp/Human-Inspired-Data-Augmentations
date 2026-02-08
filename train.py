import logging
import os

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model

from datasets.builder import build_loader
from datasets import SiamesePairDatasetBalanced, ImageNetDataset
from models import ContrastiveModel, SiameseContrastiveModel
from training import train_epoch, validate_epoch, train_epoch_siamese, validate_epoch_siamese, train_epoch_pairwise, validate_epoch_pairwise
from utils import set_seed, save_checkpoint, TripletLoss, suppress_print, suppress_wandb, suppress_logging
from utils.losses import HingeLoss, SingleTripletMultiSimilarityLoss, Oddity_Loss, TripletLoss, ContrastiveSimilarityLoss
from loguru import logger
from torch.utils.data import DataLoader
import subprocess
import os
import random

def get_loss_function(loss_cfg):
    """Create loss function based on configuration"""
    loss_type = loss_cfg.type.lower()

    if loss_type == "triplet":
        return TripletLoss(loss_cfg.margin)
    elif loss_type == "hinge":
        return HingeLoss(loss_cfg.margin)
    elif loss_type == "multi_similarity":
        return SingleTripletMultiSimilarityLoss(
            alpha=loss_cfg.get("alpha", 2),
            beta=loss_cfg.get("beta", 50),
            base=loss_cfg.get("base", 1)
        )
    elif loss_type == "oddity":
        return Oddity_Loss(temperature=loss_cfg.get("temperature", 0.1))
    elif loss_type == "pairwise_contrastive":
        return ContrastiveSimilarityLoss(temperature=loss_cfg.get("temperature", 0.5))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def is_pairwise_loss(loss_type):
    """Check if loss type requires pairwise data format"""
    return loss_type.lower() == "pairwise_contrastive"

def generate_dataset_if_needed(cfg, exp_name):
    """Generate dataset CSV if needed based on configuration"""
    dataset_type = cfg.dataset_generation.dataset_type
    
    # Determine output filename based on dataset type
    if dataset_type == "hida":
        output_filename = "hida_dataset.csv"
        script_name = "create_csv.py"
    elif dataset_type == "hida_objaverse":
        output_filename = "hida_objaverse_dataset.csv"
        script_name = "create_csv_w_Objaverse.py"
    elif dataset_type == "hida_co3d":
        output_filename = "hida_co3d_dataset.csv"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "hida_co3d_objaverse":
        output_filename = "hida_co3d_objaverse_dataset.csv"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "objaverse":
        output_filename = "objaverse_dataset.csv"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "co3d":
        output_filename = "co3d_dataset.csv"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "objaverse_co3d":
        output_filename = "objaverse_co3d_dataset.csv"
        script_name = "create_csv_w_Objaverse_co3D.py"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create experiment-specific directory and output path
    exp_dir = os.path.join("/datasets/hida/current", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    output_path = os.path.join(exp_dir, output_filename)
    
    # Check if we should regenerate (for now, always generate)
    # In production, you might want to check timestamps or add a force_regenerate flag
    
    logger.info(f"Generating dataset using {script_name}...")
    
    # Build command based on dataset type
    cmd = [
        "python", script_name,
        "--output", output_path,
        "--seed", str(cfg.dataset_generation.seed),
        "--triplets-per-combo", str(cfg.dataset_generation.triplets_per_combo)
    ]

    paths = cfg.dataset_generation.paths

    if script_name == "create_csv.py":
        # Original script - only Shapegen and Primigen
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])
    elif script_name == "create_csv_w_Objaverse.py":
        # Objaverse script - Shapegen + Primigen + Objaverse (NO CO3D support)
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])
    else:
        # CO3D script - all paths with exclusion flags
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
            "--co3d-black", paths.co3d_black,
            "--co3d-random", paths.co3d_random,
            "--co3d-white", paths.co3d_white,
            "--co3d-sim", paths.co3d_sim,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
        cmd.extend(["--co3d-ratios"] + [str(r) for r in cfg.dataset_generation.co3d_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])

        # Add dataset inclusion flags based on dataset_type
        if dataset_type == "hida_co3d":
            # Include Shapegen + Primigen + CO3D, exclude Objaverse
            cmd.extend(["--no-objaverse"])
        elif dataset_type == "hida_co3d_objaverse":
            # Include all datasets: Shapegen + Primigen + CO3D + Objaverse
            pass  # No exclusion flags needed
        elif dataset_type == "objaverse":
            # Include only Objaverse, exclude Shapegen + Primigen + CO3D
            cmd.extend(["--no-shapegen", "--no-primigen", "--no-co3d"])
        elif dataset_type == "co3d":
            # Include only CO3D, exclude Shapegen + Primigen + Objaverse
            cmd.extend(["--no-shapegen", "--no-primigen", "--no-objaverse"])
        elif dataset_type == "objaverse_co3d":
            # Include Objaverse + CO3D, exclude Shapegen + Primigen
            cmd.extend(["--no-shapegen", "--no-primigen"])

    # Run the dataset generation script
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Dataset generated successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Dataset generation failed: {e}")
        raise

    return output_path

def generate_epoch_dataset(cfg, epoch, seed, exp_name):
    """Generate a new dataset for a specific epoch"""
    dataset_type = cfg.dataset_generation.dataset_type
    
    # Determine output filename
    if dataset_type == "hida":
        base_filename = "hida_dataset"
        script_name = "create_csv.py"
    elif dataset_type == "hida_objaverse":
        base_filename = "hida_objaverse_dataset"
        script_name = "create_csv_w_Objaverse.py"
    elif dataset_type == "hida_co3d":
        base_filename = "hida_co3d_dataset"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "hida_co3d_objaverse":
        base_filename = "hida_co3d_objaverse_dataset"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "objaverse":
        base_filename = "objaverse_dataset"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "co3d":
        base_filename = "co3d_dataset"
        script_name = "create_csv_w_Objaverse_co3D.py"
    elif dataset_type == "objaverse_co3d":
        base_filename = "objaverse_co3d_dataset"
        script_name = "create_csv_w_Objaverse_co3D.py"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    output_filename = f"{base_filename}_epoch{epoch}.csv"
    # Create experiment-specific directory and output path
    exp_dir = os.path.join("/datasets/hida/current", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    output_path = os.path.join(exp_dir, output_filename)
    
    logger.info(f"Generating epoch {epoch} dataset using {script_name}...")
    
    # Build command (similar to generate_dataset_if_needed but with epoch-specific filename)
    cmd = [
        "python", script_name,
        "--output", output_path,
        "--seed", str(seed),
        "--triplets-per-combo", str(cfg.dataset_generation.triplets_per_combo)
    ]

    paths = cfg.dataset_generation.paths

    if script_name == "create_csv.py":
        # Original script - only Shapegen and Primigen
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])
    elif script_name == "create_csv_w_Objaverse.py":
        # Objaverse script - Shapegen + Primigen + Objaverse (NO CO3D support)
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])
    else:
        # CO3D script - all paths with exclusion flags
        cmd.extend([
            "--shapegen-black", paths.shapegen_black,
            "--shapegen-random", paths.shapegen_random,
            "--shapegen-white", paths.shapegen_white,
            "--shapegen-black-sim", paths.shapegen_black_sim,
            "--shapegen-random-sim", paths.shapegen_random_sim,
            "--shapegen-white-sim", paths.shapegen_white_sim,
            "--shapegen-camera", paths.shapegen_camera,
            "--primigen-black", paths.primigen_black,
            "--primigen-random", paths.primigen_random,
            "--primigen-white", paths.primigen_white,
            "--primigen-camera", paths.primigen_camera,
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
            "--co3d-black", paths.co3d_black,
            "--co3d-random", paths.co3d_random,
            "--co3d-white", paths.co3d_white,
            "--co3d-sim", paths.co3d_sim,
        ])
        cmd.extend(["--shapegen-ratios"] + [str(r) for r in cfg.dataset_generation.shapegen_ratios])
        cmd.extend(["--primigen-ratios"] + [str(r) for r in cfg.dataset_generation.primigen_ratios])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
        cmd.extend(["--co3d-ratios"] + [str(r) for r in cfg.dataset_generation.co3d_ratios])
        cmd.extend(["--n-ratios"] + [str(r) for r in cfg.dataset_generation.n_ratios])

        # Add dataset inclusion flags based on dataset_type
        if dataset_type == "hida_co3d":
            # Include Shapegen + Primigen + CO3D, exclude Objaverse
            cmd.extend(["--no-objaverse"])
        elif dataset_type == "hida_co3d_objaverse":
            # Include all datasets: Shapegen + Primigen + CO3D + Objaverse
            pass  # No exclusion flags needed
        elif dataset_type == "objaverse":
            # Include only Objaverse, exclude Shapegen + Primigen + CO3D
            cmd.extend(["--no-shapegen", "--no-primigen", "--no-co3d"])
        elif dataset_type == "co3d":
            # Include only CO3D, exclude Shapegen + Primigen + Objaverse
            cmd.extend(["--no-shapegen", "--no-primigen", "--no-objaverse"])
        elif dataset_type == "objaverse_co3d":
            # Include Objaverse + CO3D, exclude Shapegen + Primigen
            cmd.extend(["--no-shapegen", "--no-primigen"])

    # Run the dataset generation script
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Epoch {epoch} dataset generated successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Epoch {epoch} dataset generation failed: {e}")
        raise

    return output_path

def setup_ddp():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        
        # Only initialize process group if we have multiple processes
        if world_size > 1:
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, rank
    else:
        return 0, 1, 0

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    local_rank, world_size, rank = setup_ddp()
    is_main_process = rank == 0
    
    set_seed(cfg.system.random_seed)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Determine training mode
    training_mode = cfg.training.mode  # "contrastive" or "siamese"

    train_filters = cfg.splits.train_dataset.filters or {}
    val_filters = cfg.splits.val_dataset.filters or {}

    train_filter_str = '&'.join([f'{key}={value}' for key, value in train_filters.items()])
    val_filter_str = '&'.join([f'{key}={value}' for key, value in val_filters.items()])
    exp_name = f'{cfg.model.backbone.checkpoint}_bs{cfg.training.batch_size}x{world_size}_lr{cfg.optimizer.lr}_ep{cfg.training.epochs}'

    if training_mode == "siamese":
        exp_name += f'_siamese_dropout{cfg.training.siamese.dropout}_seed{cfg.system.random_seed}'
    else:
        exp_name += f'_{cfg.loss.type}_seed{cfg.system.random_seed}'

    exp_name += f'_train:{train_filter_str}_val:{val_filter_str}'
    if cfg.model.use_lora and training_mode == "contrastive":
        exp_name += f'_lora_r{cfg.model.lora_r}_alpha{cfg.model.lora_alpha}_dropout{cfg.model.lora_dropout}_|{random.randint(1, 100)}|'
    
    # Generate dataset if needed (after exp_name is created)
    dataset_csv_path = generate_dataset_if_needed(cfg, exp_name)
    
    checkpoint_dir = os.path.join(cfg.training.log_dir, exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create model based on training mode
    if training_mode == "siamese":
        model = SiameseContrastiveModel(cfg.model, cfg.training.siamese).to(device)
        backbone_status = "frozen" if cfg.training.siamese.freeze_backbone else "unfrozen"
        logger.info(f"Created Siamese model with {backbone_status} backbone")
    else:
        model = ContrastiveModel(cfg.model).to(device)

        if cfg.model.use_lora:
            lora_config = LoraConfig(
                r=cfg.model.lora_r,
                lora_alpha=cfg.model.lora_alpha,
                target_modules=["qkv", "proj"],
                lora_dropout=cfg.model.lora_dropout,
            )
            model = get_peft_model(model, lora_config)

    model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    num_workers = cfg.training.num_workers

    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone

    # Calculate train_ratio based on validation mode
    if cfg.validation.mode == "fixed":
        # Read CSV to get total dataset size
        import pandas as pd
        df = pd.read_csv(dataset_csv_path)

        # Apply filters if specified to get actual dataset size
        train_filters = cfg.splits.train_dataset.filters or {}
        if train_filters:
            for column, values in train_filters.items():
                if column in df.columns:
                    df = df[df[column] == values]

        total_size = len(df)
        fixed_val_size = cfg.validation.fixed_size

        # Ensure we have enough samples for validation
        if fixed_val_size >= total_size:
            logger.warning(f"Fixed validation size ({fixed_val_size}) >= total dataset size ({total_size}). "
                         f"Using ratio mode with default 0.95 train ratio instead.")
            calculated_train_ratio = cfg.dataset.train_ratio
        else:
            calculated_train_ratio = (total_size - fixed_val_size) / total_size
            logger.info(f"Using fixed validation size: {fixed_val_size} samples out of {total_size} total")
            logger.info(f"Calculated train_ratio: {calculated_train_ratio:.4f} (train: {total_size - fixed_val_size}, val: {fixed_val_size})")
    else:
        # Use ratio mode
        calculated_train_ratio = cfg.dataset.train_ratio
        logger.info(f"Using ratio-based validation split: train_ratio={calculated_train_ratio}")

    # Create dataset configs with updated paths and calculated ratio
    train_dataset_cfg = OmegaConf.merge(cfg.dataset, cfg.splits.train_dataset)
    train_dataset_cfg.csv_path = dataset_csv_path
    train_dataset_cfg.train_ratio = calculated_train_ratio

    val_dataset_cfg = OmegaConf.merge(cfg.dataset, cfg.splits.val_dataset)
    val_dataset_cfg.csv_path = dataset_csv_path
    val_dataset_cfg.train_ratio = calculated_train_ratio
    
    # Add augmentation config to datasets
    augmentation_config = cfg.augmentation if cfg.augmentation.enabled else None
    
    # Check if using pairwise loss (requires pair format)
    use_pairwise_data = (training_mode == "siamese") or (training_mode == "contrastive" and is_pairwise_loss(cfg.loss.type))

    # Build loaders - for Siamese mode or pairwise loss, we'll wrap the datasets after creation
    if use_pairwise_data:
        # Build triplet datasets first
        train_loader_triplet = build_loader(train_dataset_cfg, num_workers, batch_size=cfg.training.batch_size,
                                           transform=backbone.transform, num_gpus=world_size, split="train",
                                           augmentation_config=augmentation_config,
                                           augmentation_enabled=cfg.augmentation.enabled,
                                           save_every_n=cfg.augmentation.save_every_n,
                                           save_dir=cfg.augmentation.save_dir)

        val_loader_triplet = build_loader(val_dataset_cfg, num_workers, batch_size=cfg.training.batch_size,
                                         transform=backbone.transform, num_gpus=world_size, split="val",
                                         augmentation_config=None,
                                         augmentation_enabled=False)

        # Wrap triplet datasets into pair datasets
        train_pair_dataset = SiamesePairDatasetBalanced(train_loader_triplet.dataset, include_positive_negative_pairs=True)
        val_pair_dataset = SiamesePairDatasetBalanced(val_loader_triplet.dataset, include_positive_negative_pairs=True)

        # Create new dataloaders for pairs
        train_loader = DataLoader(
            train_pair_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_pair_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # For validation, we also need triplet loaders for MOCHI evaluation
        val_loader_triplet_eval = val_loader_triplet  # Keep for validation metrics
    else:
        # Contrastive mode with triplet loss - use triplet datasets directly
        train_loader = build_loader(train_dataset_cfg, num_workers, batch_size=cfg.training.batch_size,
                                   transform=backbone.transform, num_gpus=world_size, split="train",
                                   augmentation_config=augmentation_config,
                                   augmentation_enabled=cfg.augmentation.enabled,
                                   save_every_n=cfg.augmentation.save_every_n,
                                   save_dir=cfg.augmentation.save_dir)

        val_loader = build_loader(val_dataset_cfg, num_workers, batch_size=cfg.training.batch_size,
                                 transform=backbone.transform, num_gpus=world_size, split="val",
                                 augmentation_config=None,  # No augmentation for validation
                                 augmentation_enabled=False)
        val_loader_triplet_eval = val_loader  # Same as train for triplet losses
    
    mochi_loader = build_loader(cfg.splits.test_dataset, num_workers, batch_size=1,
                               transform=backbone.transform, num_gpus=1,
                               augmentation_config=None,  # No augmentation for test
                               augmentation_enabled=False)

    # Create ImageNet dataloaders if enabled
    imagenet_train_loader = None
    imagenet_test_loader = None
    imagenet_config = cfg.get('imagenet_eval', None)

    if imagenet_config and imagenet_config.get('enabled', False):
        if is_main_process:
            logger.info("Creating ImageNet evaluation dataloaders...")

        # Create train dataset first to get class directories
        imagenet_train_dataset = ImageNetDataset(
            root_dir=imagenet_config.root_dir,
            transform=backbone.transform,
            split="train",
            max_samples_per_class=imagenet_config.get('max_samples_per_class_train', 50),
            max_classes=imagenet_config.get('max_classes', 100),
            seed=cfg.system.random_seed,
            train_ratio=imagenet_config.get('train_ratio', 0.7)
        )

        # Create test dataset with same classes as train
        imagenet_test_dataset = ImageNetDataset(
            root_dir=imagenet_config.root_dir,
            transform=backbone.transform,
            split="test",
            max_samples_per_class=imagenet_config.get('max_samples_per_class_test', 20),
            max_classes=imagenet_config.get('max_classes', 100),
            seed=cfg.system.random_seed,
            class_dirs=imagenet_train_dataset.class_dirs,  # Share same classes!
            train_ratio=imagenet_config.get('train_ratio', 0.7)
        )

        imagenet_train_loader = DataLoader(
            imagenet_train_dataset,
            batch_size=imagenet_config.get('batch_size', 128),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        imagenet_test_loader = DataLoader(
            imagenet_test_dataset,
            batch_size=imagenet_config.get('batch_size', 128),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        if is_main_process:
            logger.info(f"ImageNet train dataset size: {len(imagenet_train_dataset)}")
            logger.info(f"ImageNet test dataset size: {len(imagenet_test_dataset)}")

    if is_main_process:
        # Use different wandb project for Siamese training
        wandb_project = cfg.training.siamese.wandb_project if training_mode == "siamese" else "Hida-data-size"
        wandb.init(project=wandb_project, name=exp_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.training.log_dir, entity="HIDA_dataset")

        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Using device: {device}, world_size: {world_size}")
        logger.info(f"Training mode: {training_mode}")

        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Val dataset size: {len(val_loader.dataset)}")
        logger.info(f"Mochi dataset size: {len(mochi_loader.dataset)}")

        model_for_counting = model.module if hasattr(model, 'module') else model
        logger.info(f"Model parameters: {sum(p.numel() for p in model_for_counting.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model_for_counting.parameters() if p.requires_grad):,}")

        if training_mode == "contrastive":
            logger.info(f"Loss type: {cfg.loss.type}")
            if cfg.loss.type == 'oddity':
                logger.info(f"Oddity loss temperature: {cfg.loss.temperature}")
        else:
            logger.info("Loss type: Binary Cross-Entropy (Siamese)")
            logger.info(f"Siamese dropout: {cfg.training.siamese.dropout}")
    else:
        suppress_print()
        suppress_wandb()
        suppress_logging()
    
    # Initialize loss function based on config (only for contrastive mode)
    if training_mode == "contrastive":
        criterion = get_loss_function(cfg.loss)
    else:
        criterion = None  # Siamese model has its own loss computation

    # Create optimizer with different learning rates for Siamese mode
    if training_mode == "siamese":
        # Get the actual model (unwrap DDP if needed)
        model_unwrapped = model.module if hasattr(model, 'module') else model

        # Separate parameters for backbone and siamese head
        backbone_params = list(model_unwrapped.backbone.parameters())
        siamese_params = list(model_unwrapped.siamese_net.parameters())

        # Create parameter groups with different learning rates
        param_groups = [
            {'params': siamese_params, 'lr': cfg.training.siamese.lr},  # Siamese head: configurable
        ]

        # Only add backbone params if not frozen
        if not cfg.training.siamese.freeze_backbone:
            param_groups.append({'params': backbone_params, 'lr': cfg.optimizer.lr})  # DINO backbone: uses main optimizer lr

        # Create optimizer based on selected type for Siamese mode
        optimizer_type = cfg.training.siamese.optimizer_type.lower()
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=cfg.optimizer.weight_decay,
                betas=cfg.optimizer.betas
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=cfg.optimizer.weight_decay,
                betas=cfg.optimizer.betas
            )
        else:
            raise ValueError(f"Unknown optimizer type for Siamese mode: {optimizer_type}")
    else:
        optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Create scheduler based on training mode
    if training_mode == "siamese" and cfg.training.siamese.use_scheduler:
        # Use warmup + cosine annealing scheduler for Siamese mode
        warmup_epochs = cfg.training.siamese.warmup_epochs
        max_epochs = cfg.training.epochs

        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
    else:
        # Use default CosineAnnealingWarmRestarts for contrastive mode or when scheduler disabled
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)
    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')

    # Get evaluation config
    eval_config = cfg.get('evaluation', None)

    # Initial validation
    if training_mode == "siamese":
        val_loss, val_acc, mochi_results = validate_epoch_siamese(
            model, val_loader, device, mochi_loader, is_main_process,
            imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch=0,
            eval_config=eval_config
        )
    elif is_pairwise_loss(cfg.loss.type):
        # For pairwise loss, use pair validation loader (MOCHI/ImageNet are independent)
        val_loss, val_acc, mochi_results = validate_epoch_pairwise(
            model, val_loader, criterion, device, mochi_loader, is_main_process,
            imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch=0,
            eval_config=eval_config
        )
    else:
        val_loss, val_acc, mochi_results = validate_epoch(
            model, val_loader, criterion, device, mochi_loader, is_main_process,
            imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch=0,
            eval_config=eval_config
        )
    
    for epoch in range(cfg.training.epochs):
        # Generate new dataset for each epoch if needed
        if cfg.dataset_generation.get("regenerate_per_epoch", False):
            seed = random.randint(1, 1000000)
            epoch_dataset_path = generate_epoch_dataset(cfg, epoch, seed, exp_name)

            # Recalculate train_ratio if using fixed validation mode
            if cfg.validation.mode == "fixed":
                import pandas as pd
                df = pd.read_csv(epoch_dataset_path)

                # Apply filters if specified
                train_filters = cfg.splits.train_dataset.filters or {}
                if train_filters:
                    for column, values in train_filters.items():
                        if column in df.columns:
                            df = df[df[column] == values]

                total_size = len(df)
                fixed_val_size = cfg.validation.fixed_size

                if fixed_val_size >= total_size:
                    logger.warning(f"Epoch {epoch}: Fixed validation size ({fixed_val_size}) >= total dataset size ({total_size}). "
                                 f"Using ratio mode with default 0.95 train ratio instead.")
                    epoch_train_ratio = cfg.dataset.train_ratio
                else:
                    epoch_train_ratio = (total_size - fixed_val_size) / total_size
                    if epoch == 0 or is_main_process:
                        logger.info(f"Epoch {epoch}: Recalculated train_ratio={epoch_train_ratio:.4f} "
                                  f"(train: {total_size - fixed_val_size}, val: {fixed_val_size})")
            else:
                epoch_train_ratio = calculated_train_ratio
        else:
            epoch_dataset_path = dataset_csv_path
            epoch_train_ratio = calculated_train_ratio

        # Reload dataset with updated path and ratio
        if hasattr(train_loader.dataset, "reload_data"):
            # For regular contrastive mode
            train_loader.dataset.csv_path = epoch_dataset_path
            train_loader.dataset.train_ratio = epoch_train_ratio
            train_loader.dataset.reload_data()
        elif (training_mode == "siamese" or use_pairwise_data) and hasattr(train_loader.dataset, "triplet_dataset"):
            # For Siamese mode or pairwise mode, update the underlying triplet dataset
            train_loader.dataset.triplet_dataset.csv_path = epoch_dataset_path
            train_loader.dataset.triplet_dataset.train_ratio = epoch_train_ratio
            train_loader.dataset.triplet_dataset.reload_data()

        # Update validation loader dataset if needed
        if hasattr(val_loader.dataset, "reload_data"):
            val_loader.dataset.csv_path = epoch_dataset_path
            val_loader.dataset.train_ratio = epoch_train_ratio
            val_loader.dataset.reload_data()
        elif (training_mode == "siamese" or use_pairwise_data) and hasattr(val_loader.dataset, "triplet_dataset"):
            val_loader.dataset.triplet_dataset.csv_path = epoch_dataset_path
            val_loader.dataset.triplet_dataset.train_ratio = epoch_train_ratio
            val_loader.dataset.triplet_dataset.reload_data()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        # Use appropriate training and validation functions based on mode
        if training_mode == "siamese":
            # Siamese mode
            train_epoch_siamese(model, train_loader, optimizer, device, epoch, scaler, is_main_process)
            val_loss, val_acc, mochi_results = validate_epoch_siamese(
                model, val_loader, device, mochi_loader, is_main_process,
                imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch,
                eval_config=eval_config
            )
        elif is_pairwise_loss(cfg.loss.type):
            # Contrastive mode with pairwise loss
            train_epoch_pairwise(model, train_loader, criterion, optimizer, device, epoch, scaler, is_main_process)
            # Validation uses pair loader (MOCHI/ImageNet evaluations are independent)
            val_loss, val_acc, mochi_results = validate_epoch_pairwise(
                model, val_loader, criterion, device, mochi_loader, is_main_process,
                imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch,
                eval_config=eval_config
            )
        else:
            # Contrastive mode with triplet loss
            train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, is_main_process)
            val_loss, val_acc, mochi_results = validate_epoch(
                model, val_loader, criterion, device, mochi_loader, is_main_process,
                imagenet_train_loader, imagenet_test_loader, imagenet_config, epoch,
                eval_config=eval_config
            )

        # Step scheduler after each epoch
        scheduler.step()

        # Log learning rate if in siamese mode with scheduler
        if is_main_process and training_mode == "siamese" and cfg.training.siamese.use_scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'train/learning_rate': current_lr,
                'train/epoch': epoch
            })

        if is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = save_checkpoint(model_to_save, optimizer, epoch, val_loss, checkpoint_dir)
            logger.info(f"New best model saved: {checkpoint_path}")

    if is_main_process:
        model_to_save = model.module if hasattr(model, 'module') else model
        final_checkpoint = save_checkpoint(model_to_save, optimizer, cfg.training.epochs, val_loss, checkpoint_dir)
        logger.info(f"Final model saved: {final_checkpoint}")
        wandb.finish()
        logger.info("Done :)")

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()