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
from models import ContrastiveModel
from training import train_epoch, validate_epoch
from utils import set_seed, save_checkpoint, TripletLoss, suppress_print, suppress_wandb, suppress_logging
from utils.losses import HingeLoss, SingleTripletMultiSimilarityLoss, Oddity_Loss, TripletLoss
from loguru import logger
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
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

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
    elif dataset_type == "objaverse":
        output_filename = "objaverse_dataset.csv"
        script_name = "create_csv_w_Objaverse.py"
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
    
    # Add paths based on dataset type
    if dataset_type in ["hida", "hida_objaverse"]:
        # Add Shapegen and Primigen paths
        paths = cfg.dataset_generation.paths
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
    
    if dataset_type in ["hida_objaverse", "objaverse"]:
        # Add Objaverse paths
        paths = cfg.dataset_generation.paths
        cmd.extend([
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
        ])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
    
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
    elif dataset_type == "objaverse":
        base_filename = "objaverse_dataset"
        script_name = "create_csv_w_Objaverse.py"
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
    
    # Add paths based on dataset type
    if dataset_type in ["hida", "hida_objaverse"]:
        paths = cfg.dataset_generation.paths
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
    
    if dataset_type in ["hida_objaverse", "objaverse"]:
        paths = cfg.dataset_generation.paths
        cmd.extend([
            "--objaverse-black", paths.objaverse_black,
            "--objaverse-random", paths.objaverse_random,
            "--objaverse-white", paths.objaverse_white,
            "--objaverse-sim", paths.objaverse_sim,
            ])
        cmd.extend(["--objaverse-ratios"] + [str(r) for r in cfg.dataset_generation.objaverse_ratios])
    
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
    
    train_filters = cfg.splits.train_dataset.filters or {}
    val_filters = cfg.splits.val_dataset.filters or {}
    
    train_filter_str = '&'.join([f'{key}={value}' for key, value in train_filters.items()])
    val_filter_str = '&'.join([f'{key}={value}' for key, value in val_filters.items()])
    exp_name = f'{cfg.model.backbone.checkpoint}_bs{cfg.training.batch_size}x{world_size}_lr{cfg.optimizer.lr}_ep{cfg.training.epochs}_{cfg.loss.type}_seed{cfg.system.random_seed}'
    exp_name += f'_train:{train_filter_str}_val:{val_filter_str}'
    if cfg.model.use_lora:
        exp_name += f'_lora_r{cfg.model.lora_r}_alpha{cfg.model.lora_alpha}_dropout{cfg.model.lora_dropout}_|{random.randint(1, 100)}|'
    
    # Generate dataset if needed (after exp_name is created)
    dataset_csv_path = generate_dataset_if_needed(cfg, exp_name)
    
    checkpoint_dir = os.path.join(cfg.training.log_dir, exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

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
    
    # Create dataset configs with updated paths
    train_dataset_cfg = OmegaConf.merge(cfg.dataset, cfg.splits.train_dataset)
    train_dataset_cfg.csv_path = dataset_csv_path
    
    val_dataset_cfg = OmegaConf.merge(cfg.dataset, cfg.splits.val_dataset) 
    val_dataset_cfg.csv_path = dataset_csv_path
    
    # Add augmentation config to datasets
    augmentation_config = cfg.augmentation if cfg.augmentation.enabled else None
    
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
    
    mochi_loader = build_loader(cfg.splits.test_dataset, num_workers, batch_size=cfg.evaluation.batch_size, 
                               transform=backbone.transform, num_gpus=1,
                               augmentation_config=None,  # No augmentation for test
                               augmentation_enabled=False)
    
    if is_main_process:
        wandb.init(project="hida-contrastive", name=exp_name, config=OmegaConf.to_container(cfg, resolve=True), dir=cfg.training.log_dir, entity="HIDA_dataset")

        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Using device: {device}, world_size: {world_size}")

        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Val dataset size: {len(val_loader.dataset)}")
        logger.info(f"Mochi dataset size: {len(mochi_loader.dataset)}")

        model_for_counting = model.module if hasattr(model, 'module') else model
        logger.info(f"Model parameters: {sum(p.numel() for p in model_for_counting.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model_for_counting.parameters() if p.requires_grad):,}")
        logger.info(f"Loss type: {cfg.loss.type}")
        if cfg.loss.type == 'oddity':
            logger.info(f"Oddity loss temperature: {cfg.loss.temperature}")
    else:
        suppress_print()
        suppress_wandb()
        suppress_logging()
    
    # Initialize loss function based on config
    criterion = get_loss_function(cfg.loss)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)
    scaler = torch.amp.GradScaler()

    best_val_loss = float('inf')

    val_loss, val_acc, mochi_results = validate_epoch(model, val_loader, criterion, device, mochi_loader, is_main_process)
    
    for epoch in range(cfg.training.epochs):
        # Generate new dataset for each epoch if needed
        if cfg.dataset_generation.get("regenerate_per_epoch", False):
            seed = random.randint(1, 1000000)
            epoch_dataset_path = generate_epoch_dataset(cfg, epoch, seed, exp_name)
        else:
            epoch_dataset_path = dataset_csv_path

        if hasattr(train_loader.dataset, "reload_data"):
            train_loader.dataset.csv_path = epoch_dataset_path
            train_loader.dataset.reload_data()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)

        train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, is_main_process)
        val_loss, val_acc, mochi_results = validate_epoch(model, val_loader, criterion, device, mochi_loader, is_main_process)

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