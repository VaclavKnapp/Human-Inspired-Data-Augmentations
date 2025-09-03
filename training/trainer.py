import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from evaluation.mochi_eval import evaluate_mochi
import wandb


@torch.autocast('cuda', dtype=torch.bfloat16)
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None, is_main_process=True):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=not is_main_process)
    
    for batch_idx, (anchor, positive, negative, metadata) in enumerate(pbar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        loss = criterion(anchor_emb, positive_emb, negative_emb).mean()
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #scheduler.step()
        else:
            loss.backward()
            optimizer.step()
            #scheduler.step()
        
        with torch.no_grad():
            pos_dist = 1 - F.cosine_similarity(anchor_emb, positive_emb)
            neg_dist = 1 - F.cosine_similarity(anchor_emb, negative_emb)
            correct = (pos_dist < neg_dist).sum().item()
            
        total_loss += loss.item()
        total_correct += correct
        total_samples += anchor.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/anchor.size(0):.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples

    if dist.is_initialized():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loss_tensor = torch.tensor(avg_loss, device=device)
            acc_tensor = torch.tensor(avg_acc, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
        
        avg_loss = loss_tensor.item()
        avg_acc = acc_tensor.item()

    if is_main_process:
        wandb.log({
            'train/epoch': epoch,
            'train/train_loss': avg_loss,
            'train/train_acc': avg_acc,
        })
    
    return avg_loss, avg_acc

@torch.no_grad()
@torch.autocast('cuda', dtype=torch.bfloat16)
def validate_epoch(model, val_loader, criterion, device, mochi_loader=None, is_main_process=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for anchor, positive, negative, metadata in tqdm(val_loader, desc='Validation', disable=not is_main_process):
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True) 
        negative = negative.to(device, non_blocking=True)

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb).mean()

        embs = torch.stack([anchor_emb, positive_emb, negative_emb], dim=1)
        embs_norm = F.normalize(embs, p=2, dim=-1)
        sim = torch.bmm(embs_norm, embs_norm.transpose(1, 2)).mean(dim=1)
        pred = sim.argmin(dim=1)
        correct = pred == 2

        total_loss += loss.item()
        total_correct += correct.sum()
        total_samples += anchor.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_samples

    if dist.is_initialized():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loss_tensor = torch.tensor(avg_loss, device=device)
            acc_tensor = torch.tensor(avg_acc, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
        
        avg_loss = loss_tensor.item()
        avg_acc = acc_tensor.item()
    
    mochi_results = evaluate_mochi(model, mochi_loader, device)

    if dist.is_initialized():
        for k, v in mochi_results.items():
            mochi_tensor = torch.tensor(v, device=device)
            dist.all_reduce(mochi_tensor, op=dist.ReduceOp.AVG)
            mochi_results[k] = mochi_tensor.item()

    if is_main_process:
        wandb.log({
            'val/loss': avg_loss,
            'val/acc': avg_acc,
            **mochi_results
        })
    
    return avg_loss, avg_acc, mochi_results