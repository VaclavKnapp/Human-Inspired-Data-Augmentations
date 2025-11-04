import torch
from torch.utils.data import Dataset
import random


class SiamesePairDataset(Dataset):
    """
    Wrapper dataset that converts triplet data into pairs for Siamese training.

    From each triplet (anchor, positive, negative):
    - Creates (anchor, positive) with label 1 (same object, different viewpoint)
    - Creates (anchor, negative) with label 0 (different objects)

    This effectively doubles the dataset size compared to the underlying triplet dataset.
    """

    def __init__(self, triplet_dataset):
        """
        Args:
            triplet_dataset: Dataset that returns (anchor, positive, negative, metadata)
        """
        self.triplet_dataset = triplet_dataset
        # Each triplet generates 2 pairs
        self.pairs_per_triplet = 2

    def __len__(self):
        # Each triplet creates 2 pairs
        return len(self.triplet_dataset) * self.pairs_per_triplet

    def __getitem__(self, idx):
        """
        Returns a pair of images and a label.

        Returns:
            img1: First image tensor
            img2: Second image tensor
            label: 1.0 if same object (different viewpoint), 0.0 if different objects
            metadata: Metadata dict from original triplet
        """
        # Determine which triplet to use and which pair to generate
        triplet_idx = idx // self.pairs_per_triplet
        pair_type = idx % self.pairs_per_triplet

        # Get the triplet
        anchor, positive, negative, metadata = self.triplet_dataset[triplet_idx]

        # Generate pair based on pair_type
        if pair_type == 0:
            # Positive pair: same object, different viewpoint
            img1, img2 = anchor, positive
            label = 1.0
            metadata['pair_type'] = 'same_object'
        else:
            # Negative pair: different objects
            img1, img2 = anchor, negative
            label = 0.0
            metadata['pair_type'] = 'different_object'

        return img1, img2, torch.tensor(label, dtype=torch.float32), metadata


class SiamesePairDatasetBalanced(Dataset):
    """
    Balanced version that ensures equal numbers of positive and negative pairs.
    Also adds some diversity by creating positive-negative pairs.

    From each triplet (anchor, positive, negative):
    - Creates (anchor, positive) with label 1
    - Creates (anchor, negative) with label 0
    - Optionally creates (positive, negative) with label 0
    """

    def __init__(self, triplet_dataset, include_positive_negative_pairs=True):
        """
        Args:
            triplet_dataset: Dataset that returns (anchor, positive, negative, metadata)
            include_positive_negative_pairs: If True, also create (positive, negative) pairs
        """
        self.triplet_dataset = triplet_dataset
        self.include_positive_negative_pairs = include_positive_negative_pairs
        # Each triplet generates 2 or 3 pairs
        self.pairs_per_triplet = 3 if include_positive_negative_pairs else 2

    def __len__(self):
        return len(self.triplet_dataset) * self.pairs_per_triplet

    def __getitem__(self, idx):
        """
        Returns a pair of images and a label.

        Returns:
            img1: First image tensor
            img2: Second image tensor
            label: 1.0 if same object, 0.0 if different objects
            metadata: Metadata dict from original triplet
        """
        # Determine which triplet to use and which pair to generate
        triplet_idx = idx // self.pairs_per_triplet
        pair_type = idx % self.pairs_per_triplet

        # Get the triplet
        anchor, positive, negative, metadata = self.triplet_dataset[triplet_idx]

        # Generate pair based on pair_type
        if pair_type == 0:
            # Positive pair: anchor and positive (same object, different viewpoint)
            img1, img2 = anchor, positive
            label = 1.0
            metadata['pair_type'] = 'anchor_positive'
        elif pair_type == 1:
            # Negative pair: anchor and negative (different objects)
            img1, img2 = anchor, negative
            label = 0.0
            metadata['pair_type'] = 'anchor_negative'
        else:
            # Additional negative pair: positive and negative (different objects)
            img1, img2 = positive, negative
            label = 0.0
            metadata['pair_type'] = 'positive_negative'

        return img1, img2, torch.tensor(label, dtype=torch.float32), metadata
