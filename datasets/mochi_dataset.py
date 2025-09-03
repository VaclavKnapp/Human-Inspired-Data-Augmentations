import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import ast


class MochiDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, split=None):
        self.root = "/datasets/hida/current/neurips_benchmark"
        self.data = pd.read_csv(csv_path)
        self.transform = transform

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