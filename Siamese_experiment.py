import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class EmbeddingPairDataset(Dataset):
    """Dataset for embedding pairs with optional GPU storage."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, device="cpu"):
        """Create the dataset.

        Args:
            embeddings: np.ndarray with shape (N, 2, embedding_dim)
            labels: np.ndarray with shape (N,)
            device: Target device for the returned tensors.
        """

        # Resolve the target device (fall back to CPU if CUDA not available)
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")  # gracefully degrade

        # Convert numpy arrays → torch tensors directly on the target device
        try:
            self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            self.labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        except Exception as e:
            print("Available CUDA devices:", torch.cuda.device_count())
            print("CUDA is available:", torch.cuda.is_available())
            print("torch.cuda.current_device():", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
            print("torch.cuda.get_device_name():", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A")
            print("Exception during tensor creation:", e)
            raise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class SiameseNet(nn.Module):
    """Siamese network for comparing embedding pairs."""

    def __init__(self, embedding_dim: int, dropout: float = 0.2):
        """Initialize the Siamese network.

        Args:
            embedding_dim: Dimension of input embeddings
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Shared processing tower for both embeddings
        self.shared_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 128)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x1: First embedding tensor of shape (batch_size, embedding_dim)
            x2: Second embedding tensor of shape (batch_size, embedding_dim)

        Returns:
            Logits for binary classification (before sigmoid)
        """
        # Process both embeddings through shared tower
        e1 = F.normalize(self.shared_tower(x1), p=2, dim=-1)
        e2 = F.normalize(self.shared_tower(x2), p=2, dim=-1)

        # Compute absolute difference
        diff = torch.abs(e1 - e2)

        # Classify (return logits, not probabilities)
        logits = self.classifier(diff).squeeze(-1)
        return logits

    def compute_loss(self, x1: torch.Tensor, x2: torch.Tensor,
                     labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and accuracy for a batch.

        Args:
            x1: First embedding tensor
            x2: Second embedding tensor
            labels: Ground truth labels (0 or 1)

        Returns:
            Tuple of (loss, accuracy)
        """
        logits = self(x1, x2)
        #autocast-safe
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Calculate accuracy (apply sigmoid for prediction)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()

        return loss, acc
