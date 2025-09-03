import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import random
import pandas as pd
import os
import io
import shutil
from loguru import logger

class AddGaussianNoise:
    """Adds random Gaussian noise to a tensor image."""
    def __init__(self, mean=0., std_min=0.003, std_max=0.03):
        self.mean = mean
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, tensor):
        std = random.uniform(self.std_min, self.std_max)
        noise = torch.randn(tensor.size()) * std + self.mean
        return torch.clamp(tensor + noise, 0, 1)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std_min={self.std_min}, std_max={self.std_max})"

class AddMotionBlur:
    """Applies motion blur with a random angle."""
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __call__(self, tensor):
        angle = random.uniform(0, 180)
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = (self.kernel_size - 1) / 2
        
        radians = np.radians(angle)
        x_end = center + center * np.cos(radians)
        y_end = center + center * np.sin(radians)
        
        # Use OpenCV's line drawing if available, otherwise fallback
        try:
            import cv2
            cv2.line(kernel, (int(center - x_end), int(center - y_end)), (int(center + x_end), int(center + y_end)), 1.0, 1)
        except ImportError:
            # Fallback if OpenCV is not installed
            for i in range(self.kernel_size):
                y = int(np.tan(radians) * (i - center) + center)
                if 0 <= y < self.kernel_size:
                    kernel[y, i] = 1.0
        
        kernel = torch.from_numpy(kernel).float()
        kernel /= kernel.sum()

        channels = tensor.shape[0]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(channels, 1, 1, 1)
        padding = self.kernel_size // 2
        blurred_tensor = F.conv2d(tensor.unsqueeze(0), kernel, padding=padding, groups=channels)
        return blurred_tensor.squeeze(0)

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size})"


class DomainRandomizationTransform:
    """Domain randomization transform that applies augmentations based on configuration.
    Normalization will be handled by the backbone transform."""

    def __init__(self, augmentation_config=None):
        if augmentation_config is None:
            # Default configuration for backwards compatibility
            augmentation_config = {
                'color_jitter': {
                    'brightness': 0.10, 'contrast': 0.13, 
                    'saturation': 0.1, 'hue': 0.05, 'probability': 0.5
                },
                'gaussian_blur': {
                    'kernel_size': 3, 'sigma': [0.1, 0.6], 'probability': 0.5
                },
                'motion_blur': {
                    'kernel_size': 5, 'probability': 0.5
                },
                'gaussian_noise': {
                    'mean': 0.0, 'std_min': 0.003, 'std_max': 0.03, 'probability': 0.5
                }
            }
        
        augmentations = []
        
        # Color Jitter
        if 'color_jitter' in augmentation_config:
            cj_cfg = augmentation_config['color_jitter']
            augmentations.append(
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=cj_cfg.get('brightness', 0.1),
                        contrast=cj_cfg.get('contrast', 0.1),
                        saturation=cj_cfg.get('saturation', 0.1),
                        hue=cj_cfg.get('hue', 0.05)
                    )
                ], p=cj_cfg.get('probability', 0.5))
            )
        
        # Gaussian Blur
        if 'gaussian_blur' in augmentation_config:
            gb_cfg = augmentation_config['gaussian_blur']
            augmentations.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=gb_cfg.get('kernel_size', 3),
                        sigma=gb_cfg.get('sigma', [0.1, 0.6])
                    )
                ], p=gb_cfg.get('probability', 0.5))
            )
        
        # Motion Blur
        if 'motion_blur' in augmentation_config:
            mb_cfg = augmentation_config['motion_blur']
            augmentations.append(
                transforms.RandomApply([
                    AddMotionBlur(kernel_size=mb_cfg.get('kernel_size', 5))
                ], p=mb_cfg.get('probability', 0.5))
            )
        
        # Gaussian Noise
        if 'gaussian_noise' in augmentation_config:
            gn_cfg = augmentation_config['gaussian_noise']
            augmentations.append(
                transforms.RandomApply([
                    AddGaussianNoise(
                        mean=gn_cfg.get('mean', 0.0),
                        std_min=gn_cfg.get('std_min', 0.003),
                        std_max=gn_cfg.get('std_max', 0.03)
                    )
                ], p=gn_cfg.get('probability', 0.5))
            )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomOrder(augmentations)
        ])

    def __call__(self, image):
        return self.transform(image)


class HIDADataset(Dataset):
    """
    Loads triplets (A, A', B), applies domain randomization, and saves a
    sample of augmented images to disk for inspection.
    """
    def __init__(self, csv_path, transform=None, split=None, filters=None, 
                 train_ratio=0.8, random_seed=42, save_every_n=1000, save_dir="./augmented_samples",
                 augmentation_config=None, augmentation_enabled=True):
        self.csv_path = csv_path
        self.filters = filters
        self.split = split
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.data = pd.read_csv(self.csv_path)
        self.split = split
        self.augmentations = augmentation_enabled
        
        # Apply filters if provided
        if filters:
            self.data = self.apply_filters(self.data, filters)
        
        # Apply train/val split if provided
        if split is not None:
            self.data = self.apply_split(self.data, split, train_ratio, random_seed)
            
        # Set up transforms
        self.augmentation_transform = DomainRandomizationTransform(augmentation_config)
        self.backbone_transform = transform  # This is the backbone's transform (from dinov2)
        
        self.save_every_n = save_every_n
        self.save_dir = save_dir
        
        if split == 'train' and self.augmentations:
            print("AUGMENTATIONS ENABLED FOR TRAINING...")
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)
                print(f"Augmented samples will be saved to '{self.save_dir}' every {self.save_every_n} items.")
        else:
            print(f"NO AUGMENTATIONS FOR {split.upper() if split else 'INFERENCE'} - using backbone transform only")

    def reload_data(self, csv_path=None):
        """Reload CSV, reapply filters & split."""
        path = csv_path or self.csv_path
        df = pd.read_csv(path)
        if self.filters:
            df = self.apply_filters(df, self.filters)
        if self.split is not None:
            df = self.apply_split(df, self.split, self.train_ratio, self.random_seed)
        self.data = df.reset_index(drop=True)

    def apply_filters(self, data, filters):
        for column, values in filters.items():
            if column in data.columns:
                data = data[data[column] == values]
                logger.info(f"Filtered by {column}={values}: {len(data)} samples remaining")
            else:
                logger.warning(f"Column '{column}' not found in dataset")
        
        return data.reset_index(drop=True)

    def get_by_trial(self, trial):
        trial_data = self.data[self.data['Trial'] == trial]
        return self.__getitem__(trial_data.index.item())

    def apply_split(self, data, split, train_ratio, random_seed):
        np.random.seed(random_seed)
        
        n_samples = len(data)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_size = int(n_samples * train_ratio)
        
        if split == 'train':
            selected_indices = indices[:train_size]
            logger.info(f"Using train split: {len(selected_indices)} samples")
        elif split == 'val':
            selected_indices = indices[train_size:]
            logger.info(f"Using val split: {len(selected_indices)} samples") 
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        return data.iloc[selected_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            anchor_img = self.load_image(row['A'])
            positive_img = self.load_image(row['A_prime'])
            negative_img = self.load_image(row['B'])
        except FileNotFoundError as e:
            print(f"Warning: File not found for row {idx}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))
            
        # Apply augmentations only during training
        if self.split == 'train' and self.augmentations:
            # Apply domain randomization augmentations first
            anchor_tensor = self.augmentation_transform(anchor_img)
            positive_tensor = self.augmentation_transform(positive_img)
            negative_tensor = self.augmentation_transform(negative_img)
            

            anchor_img = transforms.ToPILImage()(anchor_tensor)
            positive_img = transforms.ToPILImage()(positive_tensor)
            negative_img = transforms.ToPILImage()(negative_tensor)
            
            # Save augmented images for inspection
            if self.save_dir and self.save_every_n > 0 and idx % self.save_every_n == 0:
                trial_id = row.get('Trial', idx)
                self._save_tensor_for_inspection(anchor_tensor, trial_id, idx, 'anchor')
                self._save_tensor_for_inspection(positive_tensor, trial_id, idx, 'positive')
                self._save_tensor_for_inspection(negative_tensor, trial_id, idx, 'negative')
        
        # Apply backbone transform (includes normalization)
        if self.backbone_transform:
            anchor = self.backbone_transform(anchor_img)
            positive = self.backbone_transform(positive_img)
            negative = self.backbone_transform(negative_img)
        else:
            # Fallback if no backbone transform
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            anchor = to_tensor(anchor_img)
            positive = to_tensor(positive_img)
            negative = to_tensor(negative_img)
            
        metadata = {
            'trial': row.get('Trial', -1),
            'bg': row.get('BG', ''),
            'dataset': row.get('DATASET', ''),
            'condition': row.get('CONDITION', ''),
            'name_a': row.get('NAME_A', ''),
            'name_b': row.get('NAME_B', '')
        }
        
        return anchor, positive, negative, metadata
    
    def _save_tensor_for_inspection(self, tensor, trial_id, idx, image_type):
        """Save tensor for inspection. Assumes tensor is in [0,1] range (not normalized)."""
        img_to_save = torch.clamp(tensor, 0, 1)
        
        filename = f"trial_{trial_id}_idx_{idx}_{image_type}.png"
        save_path = os.path.join(self.save_dir, filename)
        
        save_image(img_to_save, save_path)

    def load_image(self, path):
        return Image.open(path).convert('RGB')