import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import warnings

from evaluation.mochi_eval import evaluate_mochi, evaluate_mochi_siamese
from evaluation.imagenet_eval import evaluate_imagenet
import wandb
import torch.nn.functional as F_func


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


@torch.autocast('cuda', dtype=torch.bfloat16)
def train_epoch_pairwise(model, train_loader, criterion, optimizer, device, epoch, scaler=None, is_main_process=True):
    """
    Training epoch for pairwise contrastive loss.
    Expects train_loader to return (img1, img2, label, metadata).
    - label=1.0 for similar pairs
    - label=0.0 for dissimilar pairs
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Pairwise]', disable=not is_main_process)

    for batch_idx, (img1, img2, labels, metadata) in enumerate(pbar):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device).float()  # Ensure float for BCE

        optimizer.zero_grad()

        # Extract embeddings
        emb1 = model(img1)
        emb2 = model(img2)

        # Compute pairwise loss
        loss = criterion(emb1, emb2, labels).mean()

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Compute accuracy: predict similarity based on cosine similarity
        with torch.no_grad():
            emb1_norm = F_func.normalize(emb1, p=2, dim=-1)
            emb2_norm = F_func.normalize(emb2, p=2, dim=-1)
            similarity = (emb1_norm * emb2_norm).sum(dim=-1)
            predicted = (similarity > 0.0).float()  # Threshold at 0 (cosine similarity range [-1, 1])
            correct = (predicted == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += img1.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/img1.size(0):.4f}'
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
def validate_epoch(model, val_loader, criterion, device, mochi_loader=None, is_main_process=True,
                   imagenet_train_loader=None, imagenet_test_loader=None, imagenet_config=None, epoch=0,
                   eval_config=None):
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

    # MOCHI evaluation with configurable components
    mochi_results = {}
    if eval_config:
        run_mochi_cosine = eval_config.get('run_mochi_cosine', True)
        run_mochi_svm = eval_config.get('run_mochi_svm', True)
    else:
        run_mochi_cosine = True
        run_mochi_svm = True

    if run_mochi_cosine or run_mochi_svm:
        mochi_results = evaluate_mochi(model, mochi_loader, device,
                                      run_cosine=run_mochi_cosine,
                                      run_svm=run_mochi_svm)

        if dist.is_initialized():
            for k, v in mochi_results.items():
                mochi_tensor = torch.tensor(v, device=device)
                dist.all_reduce(mochi_tensor, op=dist.ReduceOp.AVG)
                mochi_results[k] = mochi_tensor.item()

    # ImageNet evaluation
    imagenet_results = {}
    run_imagenet = eval_config.get('run_imagenet', True) if eval_config else True

    if run_imagenet and imagenet_config and imagenet_config.get('enabled', False):
        eval_freq = imagenet_config.get('eval_frequency', 5)
        if epoch % eval_freq == 0 and imagenet_train_loader is not None and imagenet_test_loader is not None:
            if is_main_process:
                print(f"\nRunning ImageNet evaluation at epoch {epoch}...")
            imagenet_results = evaluate_imagenet(
                model,
                imagenet_train_loader,
                imagenet_test_loader,
                device,
                knn_k=imagenet_config.get('knn_k', 20),
                eval_knn=imagenet_config.get('eval_knn', True),
                eval_linear=imagenet_config.get('eval_linear', False),
                is_main_process=is_main_process
            )

    if is_main_process:
        wandb.log({
            'val/loss': avg_loss,
            'val/acc': avg_acc,
            **mochi_results,
            **imagenet_results
        })

    return avg_loss, avg_acc, {**mochi_results, **imagenet_results}


@torch.no_grad()
@torch.autocast('cuda', dtype=torch.bfloat16)
def validate_epoch_pairwise(model, val_loader, criterion, device, mochi_loader=None, is_main_process=True,
                            imagenet_train_loader=None, imagenet_test_loader=None, imagenet_config=None, epoch=0,
                            eval_config=None):
    """
    Validation epoch for pairwise contrastive loss.
    Expects val_loader to return (img1, img2, label, metadata).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for img1, img2, labels, metadata in tqdm(val_loader, desc='Validation [Pairwise]', disable=not is_main_process):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        # Extract embeddings
        emb1 = model(img1)
        emb2 = model(img2)

        # Compute pairwise loss
        loss = criterion(emb1, emb2, labels).mean()

        # Compute accuracy
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        similarity = (emb1_norm * emb2_norm).sum(dim=-1)
        predicted = (similarity > 0.0).float()
        correct = (predicted == labels).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += img1.size(0)

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

    # MOCHI evaluation (independent of loss type)
    mochi_results = {}
    if eval_config:
        run_mochi_cosine = eval_config.get('run_mochi_cosine', True)
        run_mochi_svm = eval_config.get('run_mochi_svm', True)
    else:
        run_mochi_cosine = True
        run_mochi_svm = True

    if run_mochi_cosine or run_mochi_svm:
        mochi_results = evaluate_mochi(model, mochi_loader, device,
                                      run_cosine=run_mochi_cosine,
                                      run_svm=run_mochi_svm)

        if dist.is_initialized():
            for k, v in mochi_results.items():
                mochi_tensor = torch.tensor(v, device=device)
                dist.all_reduce(mochi_tensor, op=dist.ReduceOp.AVG)
                mochi_results[k] = mochi_tensor.item()

    # ImageNet evaluation (independent of loss type)
    imagenet_results = {}
    run_imagenet = eval_config.get('run_imagenet', True) if eval_config else True

    if run_imagenet and imagenet_config and imagenet_config.get('enabled', False):
        eval_freq = imagenet_config.get('eval_frequency', 5)
        if epoch % eval_freq == 0 and imagenet_train_loader is not None and imagenet_test_loader is not None:
            if is_main_process:
                print(f"\nRunning ImageNet evaluation at epoch {epoch}...")
            imagenet_results = evaluate_imagenet(
                model,
                imagenet_train_loader,
                imagenet_test_loader,
                device,
                knn_k=imagenet_config.get('knn_k', 20),
                eval_knn=imagenet_config.get('eval_knn', True),
                eval_linear=imagenet_config.get('eval_linear', False),
                is_main_process=is_main_process
            )

    if is_main_process:
        wandb.log({
            'val/loss': avg_loss,
            'val/acc': avg_acc,
            **mochi_results,
            **imagenet_results
        })

    return avg_loss, avg_acc, {**mochi_results, **imagenet_results}


@torch.autocast('cuda', dtype=torch.bfloat16)
def train_epoch_siamese(model, train_loader, optimizer, device, epoch, scaler=None, is_main_process=True):
    """
    Training epoch for Siamese network.
    Expects train_loader to return (img1, img2, label, metadata).
    """
    model.train()

    # Set backbone to eval mode if it's frozen
    if hasattr(model, 'siamese_cfg') and model.siamese_cfg.freeze_backbone:
        model.backbone.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Siamese]', disable=not is_main_process)

    for batch_idx, (img1, img2, labels, metadata) in enumerate(pbar):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Compute loss using model's compute_loss method
        loss, acc = model.compute_loss(img1, img2, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_correct += acc.item() * img1.size(0)
        total_samples += img1.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc.item():.4f}'
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
def validate_epoch_siamese(model, val_loader, device, mochi_loader=None, is_main_process=True,
                          imagenet_train_loader=None, imagenet_test_loader=None, imagenet_config=None, epoch=0,
                          eval_config=None):
    """
    Validation epoch for Siamese network.
    Expects val_loader to return (img1, img2, label, metadata).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for img1, img2, labels, metadata in tqdm(val_loader, desc='Validation [Siamese]', disable=not is_main_process):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Compute loss using model's compute_loss method
        loss, acc = model.compute_loss(img1, img2, labels)

        total_loss += loss.item()
        total_correct += acc.item() * img1.size(0)
        total_samples += img1.size(0)

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

    # Evaluate on Mochi with Siamese-specific logic
    # Note: For Siamese, we only use the siamese-specific evaluation, not the cosine/SVM
    mochi_results = evaluate_mochi_siamese(model, mochi_loader, device)

    if dist.is_initialized():
        for k, v in mochi_results.items():
            mochi_tensor = torch.tensor(v, device=device)
            dist.all_reduce(mochi_tensor, op=dist.ReduceOp.AVG)
            mochi_results[k] = mochi_tensor.item()

    # ImageNet evaluation (using backbone only for Siamese model)
    imagenet_results = {}
    run_imagenet = eval_config.get('run_imagenet', True) if eval_config else True

    if run_imagenet and imagenet_config and imagenet_config.get('enabled', False):
        eval_freq = imagenet_config.get('eval_frequency', 5)
        if epoch % eval_freq == 0 and imagenet_train_loader is not None and imagenet_test_loader is not None:
            if is_main_process:
                print(f"\nRunning ImageNet evaluation at epoch {epoch}...")
            # For Siamese model, evaluate using backbone only
            backbone_model = model.backbone if hasattr(model, 'backbone') else model
            imagenet_results = evaluate_imagenet(
                backbone_model,
                imagenet_train_loader,
                imagenet_test_loader,
                device,
                knn_k=imagenet_config.get('knn_k', 20),
                eval_knn=imagenet_config.get('eval_knn', True),
                eval_linear=imagenet_config.get('eval_linear', False),
                is_main_process=is_main_process
            )

    if is_main_process:
        wandb.log({
            'val/loss': avg_loss,
            'val/acc': avg_acc,
            **mochi_results,
            **imagenet_results
        })

    return avg_loss, avg_acc, {**mochi_results, **imagenet_results}