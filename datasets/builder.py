import os
import torch
from hydra.utils import instantiate
from PIL import ImageFile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")


def build_loader(cfg, n_workers, batch_size, num_gpus=1, split=None, **kwargs):
    # Filter kwargs based on the target dataset class
    if cfg._target_ == "datasets.mochi_dataset.MochiDataset":
        # MochiDataset only accepts csv_path, transform, and split
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['transform']}
    else:
        # HIDADataset accepts all kwargs
        filtered_kwargs = kwargs
    
    dataset = instantiate(cfg, split=split, **filtered_kwargs)
    
    use_ddp = num_gpus > 1
    sampler = DistributedSampler(dataset) if use_ddp else None
    shuffle = (split == "train") and not use_ddp
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, drop_last=True if split == "train" else False, pin_memory=True, shuffle=shuffle, sampler=sampler)
    
    return loader


def build_dataset(cfg, split=None, **kwargs):
    # Filter kwargs based on the target dataset class
    if cfg._target_ == "datasets.mochi_dataset.MochiDataset":
        # MochiDataset only accepts csv_path, transform, and split
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in ['transform']}
    else:
        # HIDADataset accepts all kwargs
        filtered_kwargs = kwargs
    
    return instantiate(cfg, split=split, **filtered_kwargs)