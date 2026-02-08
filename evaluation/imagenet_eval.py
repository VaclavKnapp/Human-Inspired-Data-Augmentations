import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict


@torch.no_grad()
def extract_features(model, dataloader, device, is_main_process=True):
    """
    Extract features and labels from a dataset using the model.

    Args:
        model: Feature extraction model
        dataloader: DataLoader for the dataset
        device: Device to run on
        is_main_process: Whether this is the main process (for logging)

    Returns:
        features: Numpy array of features (N, D)
        labels: Numpy array of labels (N,)
    """
    model.eval()

    all_features = []
    all_labels = []

    desc = "Extracting ImageNet features"
    for images, labels in tqdm(dataloader, desc=desc, disable=not is_main_process):
        images = images.to(device, non_blocking=True)

        # Extract features
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features = model(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

    # Concatenate all batches
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return features, labels


@torch.no_grad()
def evaluate_knn(
    model,
    train_loader,
    test_loader,
    device,
    k: int = 20,
    temperature: float = 0.07,
    is_main_process: bool = True
):
    """
    Evaluate model using k-NN classification.

    Args:
        model: Feature extraction model
        train_loader: DataLoader for training set (features are extracted)
        test_loader: DataLoader for test set
        device: Device to run on
        k: Number of neighbors for k-NN
        temperature: Temperature for cosine similarity
        is_main_process: Whether this is the main process

    Returns:
        Dictionary with top-1 and top-5 accuracy
    """
    if is_main_process:
        print(f"\n{'='*50}")
        print(f"ImageNet k-NN Evaluation (k={k})")
        print(f"{'='*50}")

    # Extract training features
    train_features, train_labels = extract_features(model, train_loader, device, is_main_process)

    # Extract test features
    test_features, test_labels = extract_features(model, test_loader, device, is_main_process)

    if is_main_process:
        print(f"Train features: {train_features.shape}, Test features: {test_features.shape}")

    # Normalize features for cosine similarity
    train_features = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
    test_features = test_features / (np.linalg.norm(test_features, axis=1, keepdims=True) + 1e-8)

    # Use sklearn KNN for efficient computation
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
    knn.fit(train_features, train_labels)

    # Get predictions
    # For top-k accuracy, we need the probabilities/distances
    distances, indices = knn.kneighbors(test_features)

    # Get top-1 predictions
    top1_preds = knn.predict(test_features)
    top1_acc = (top1_preds == test_labels).mean()

    # For top-5, we look at the k nearest neighbors
    # Get the labels of k nearest neighbors
    neighbor_labels = train_labels[indices]  # (N_test, k)

    # Top-5: check if true label is in top 5 most common labels among k neighbors
    top5_correct = 0
    for i, true_label in enumerate(test_labels):
        # Count label occurrences among neighbors
        unique, counts = np.unique(neighbor_labels[i], return_counts=True)
        # Get top 5 most common labels
        top5_labels = unique[np.argsort(-counts)[:5]]
        if true_label in top5_labels:
            top5_correct += 1

    top5_acc = top5_correct / len(test_labels)

    if is_main_process:
        print(f"k-NN Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"k-NN Top-5 Accuracy: {top5_acc*100:.2f}%")

    return {
        'imagenet_knn_top1': top1_acc,
        'imagenet_knn_top5': top5_acc,
    }


@torch.no_grad()
def evaluate_linear_probe(
    model,
    train_loader,
    test_loader,
    device,
    max_iter: int = 1000,
    is_main_process: bool = True
):
    """
    Evaluate model using linear probe (logistic regression on frozen features).

    Args:
        model: Feature extraction model
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        device: Device to run on
        max_iter: Maximum iterations for logistic regression
        is_main_process: Whether this is the main process

    Returns:
        Dictionary with top-1 accuracy
    """
    if is_main_process:
        print(f"\n{'='*50}")
        print(f"ImageNet Linear Probe Evaluation")
        print(f"{'='*50}")

    # Extract training features
    train_features, train_labels = extract_features(model, train_loader, device, is_main_process)

    # Extract test features
    test_features, test_labels = extract_features(model, test_loader, device, is_main_process)

    if is_main_process:
        print(f"Train features: {train_features.shape}, Test features: {test_features.shape}")
        print(f"Training linear classifier...")

    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Train logistic regression
    classifier = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs',
        multi_class='multinomial',
        verbose=1 if is_main_process else 0,
        n_jobs=-1
    )
    classifier.fit(train_features, train_labels)

    # Evaluate
    top1_acc = classifier.score(test_features, test_labels)

    if is_main_process:
        print(f"Linear Probe Top-1 Accuracy: {top1_acc*100:.2f}%")

    return {
        'imagenet_linear_top1': top1_acc,
    }


@torch.no_grad()
def evaluate_imagenet(
    model,
    train_loader,
    test_loader,
    device,
    knn_k: int = 20,
    eval_knn: bool = True,
    eval_linear: bool = False,
    is_main_process: bool = True
):
    """
    Comprehensive ImageNet evaluation with both k-NN and linear probe.

    Args:
        model: Feature extraction model
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        device: Device to run on
        knn_k: Number of neighbors for k-NN
        eval_knn: Whether to evaluate k-NN
        eval_linear: Whether to evaluate linear probe
        is_main_process: Whether this is the main process

    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()

    results = {}

    # k-NN evaluation
    if eval_knn:
        knn_results = evaluate_knn(model, train_loader, test_loader, device, k=knn_k, is_main_process=is_main_process)
        results.update(knn_results)

    # Linear probe evaluation
    if eval_linear:
        linear_results = evaluate_linear_probe(model, train_loader, test_loader, device, is_main_process=is_main_process)
        results.update(linear_results)

    return results
