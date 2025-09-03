import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict

from utils.training_utils import adjust_accuracy


@torch.no_grad()
def evaluate_mochi(model, mochi_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    acc_per_condition = defaultdict(float)
    total_per_condition = defaultdict(int)
    n_images_per_condition = dict()

    for images, dataset, condition, oddity_indices in tqdm(mochi_loader, desc='Mochi Evaluation'):
        images = images.to(device)
        b_actual, n_images, c, h, w = images.shape
        images_flat = images.view(-1, c, h, w)

        features = model(images_flat)
        features = features.view(b_actual, n_images, -1)

        features_norm = features / features.norm(dim=2, keepdim=True)
        sim_matrices = torch.bmm(features_norm, features_norm.transpose(1, 2))

        diagonal_mask = torch.eye(n_images, device=device).unsqueeze(0).expand(b_actual, -1, -1)
        sim_matrices_no_diag = sim_matrices - diagonal_mask
        sim_mean = sim_matrices_no_diag.sum(dim=2) / (n_images - 1)
        pred_oddity = torch.argmin(sim_mean, dim=1)

        correct = (pred_oddity == oddity_indices.to(device)).float()
        acc = adjust_accuracy(correct, 1/n_images)
        
        total_correct += acc.sum().item()
        total_samples += b_actual

        for i, (dataset_i, condition_i) in enumerate(zip(dataset, condition)):
            cond_name = f"{dataset_i}_{condition_i}"
            n_images_per_condition[cond_name] = n_images
            acc_per_condition[cond_name] += acc[i].item()
            total_per_condition[cond_name] += 1

    accuracy = total_correct / total_samples

    if dist.is_initialized():
        correct_tensor = torch.tensor(total_correct, device=device)
        samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        
        accuracy = correct_tensor.item() / samples_tensor.item()

        for cond_name in acc_per_condition.keys():
            cond_correct = torch.tensor(acc_per_condition[cond_name], device=device)
            cond_total = torch.tensor(total_per_condition[cond_name], device=device)
            
            dist.all_reduce(cond_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(cond_total, op=dist.ReduceOp.SUM)
            
            acc_per_condition[cond_name] = cond_correct.item() / cond_total.item()
    else:
        acc_per_condition = {k: v / total_per_condition[k] for k, v in acc_per_condition.items()}
    
    metrics = {'mochi_overall': accuracy}

    for condition, acc in acc_per_condition.items():
        metrics[f'mochi/{condition}'] = acc
    return metrics
