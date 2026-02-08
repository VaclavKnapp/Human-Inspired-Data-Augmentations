import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageNetDataset(Dataset):
    """
    ImageNet dataset for evaluation.
    Expects directory structure: root_dir/class_id/image.JPEG
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        split: str = "train",
        max_samples_per_class: Optional[int] = None,
        max_classes: Optional[int] = None,
        seed: int = 42,
        class_dirs: Optional[List] = None,
        train_ratio: float = 0.7
    ):
        """
        Args:
            root_dir: Root directory containing class folders
            transform: Transform to apply to images
            split: Split name ('train' or 'test')
            max_samples_per_class: Maximum samples to use per class (for subset evaluation)
            max_classes: Maximum number of classes to use (for subset evaluation)
            seed: Random seed for sampling
            class_dirs: Optional list of class directories to use (for sharing classes between train/test)
            train_ratio: Ratio of samples to use for training (rest go to test)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.seed = seed
        self.train_ratio = train_ratio

        # Get class directories (either provided or discovered)
        if class_dirs is not None:
            # Use provided class directories
            class_dirs = class_dirs
        else:
            # Get all class directories
            class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

            # Optionally limit number of classes
            if max_classes is not None and max_classes < len(class_dirs):
                np.random.seed(seed)
                class_dirs = np.random.choice(class_dirs, size=max_classes, replace=False).tolist()
                class_dirs = sorted(class_dirs)

        # Store for potential reuse
        self.class_dirs = class_dirs

        # Build class to index mapping
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            # Get all images in this class
            images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            # Shuffle images deterministically
            np.random.seed(seed + class_idx)
            images = np.array(images)
            np.random.shuffle(images)
            images = images.tolist()

            # Split into train/test based on split parameter
            if split == "train":
                # Take first train_ratio portion
                n_train = int(len(images) * train_ratio)
                if max_samples_per_class is not None:
                    n_train = min(n_train, max_samples_per_class)
                images = images[:n_train]
            elif split == "test":
                # Take remaining portion
                n_train = int(len(images) * train_ratio)
                images = images[n_train:]
                if max_samples_per_class is not None:
                    images = images[:max_samples_per_class]

            for img_path in images:
                self.samples.append((str(img_path), class_idx))

        print(f"ImageNet {split}: {len(self.classes)} classes, {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label
