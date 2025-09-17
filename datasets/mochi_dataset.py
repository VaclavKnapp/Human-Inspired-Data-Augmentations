import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import ast
import numpy as np


class MochiDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, split=None,
                 sample_percentage=None, sample_seed=42):
        self.root = os.path.dirname(csv_path)
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Apply percentage sampling if specified
        if sample_percentage is not None and 0 < sample_percentage < 1:
            self.data = self._sample_by_category(self.data, sample_percentage, sample_seed)

    def _sample_by_category(self, data, percentage, seed):
        """Sample a percentage of data from each dataset-condition combination"""
        np.random.seed(seed)

        # Group by dataset and condition
        grouped = data.groupby(['dataset', 'condition'])
        sampled_groups = []

        for (dataset, condition), group in grouped:
            n_samples = len(group)
            n_to_sample = max(1, int(n_samples * percentage))  # At least 1 sample

            if n_to_sample >= n_samples:
                sampled_groups.append(group)
            else:
                sampled_idx = np.random.choice(group.index, size=n_to_sample, replace=False)
                sampled_groups.append(group.loc[sampled_idx])

        # Concatenate all sampled groups and reset index
        sampled_data = pd.concat(sampled_groups).reset_index(drop=True)
        return sampled_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        images = []
        for img_path in ast.literal_eval(row['images']):
            img = Image.open(os.path.join(self.root, 'images', img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)
        oddity_index = row['oddity_index']
        condition = row['condition']
        dataset = row['dataset']

        return images, dataset, condition, oddity_index