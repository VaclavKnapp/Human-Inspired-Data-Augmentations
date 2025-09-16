import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from utils.training_utils import adjust_accuracy
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def _pair_diffs_and_labels(feats_n, odd_idx):
    """
    feats_n: (n_images, D) tensor (CPU or CUDA)
    odd_idx: int in {0, 1, ..., n_images-1}
    Returns:
      X: (n_pairs, D) torch.Tensor of pairwise diffs
      y: (n_pairs,) torch.Tensor of labels (same=1, different=0)
      pairs: list of index pairs corresponding to rows of X
    """
    n_images = feats_n.shape[0]
    idxs = list(range(n_images))
    non_odd_idxs = [i for i in idxs if i != odd_idx]  # all non-odd images
    
    X = []
    y = []
    pairs = []

    def diff(i, j):
        i1, j1 = (i, j) if i < j else (j, i)
        return feats_n[i1] - feats_n[j1], (i1, j1)

    # Generate all possible pairs and label them
    for i in range(n_images):
        for j in range(i + 1, n_images):
            d, p = diff(i, j)
            X.append(d)
            pairs.append(p)
            
            # Label: 1 if both are non-odd (same), 0 if one is odd (different)
            if i != odd_idx and j != odd_idx:
                y.append(1)  # both are typical/same
            else:
                y.append(0)  # one is odd/different

    X = torch.stack(X, dim=0)
    y = torch.tensor(y, dtype=torch.long, device=X.device)
    return X, y, pairs

@torch.no_grad()
def evaluate_mochi(
    model,
    mochi_loader,
    device,
    svm_repeats: int = 50,  # n_permutations in reference
    train_fraction: float = 0.75,  # subset_fraction in reference
    svm_C: float = 1.0,
    svm_max_iter: int = 5000,
):
    """
    Computes BOTH:
      1) cosine-argmin oddity (zero-shot)
      2) same–different linear SVM readout (per-condition, leave-one-triplet-out with repeated resampling)

    Returns a dict with:
      - mochi_cosine_overall, mochi_samediff_overall
      - per-condition: mochi_cosine/{dataset_condition}, mochi_samediff/{dataset_condition}
    """

    model.eval()

    total_correct_cos = 0.0
    total_samples = 0
    acc_per_condition_cos = defaultdict(float)
    total_per_condition = defaultdict(int)
    n_images_per_condition = dict()

    # cache for SVM readout: {cond_name: list of dicts with 'feats' (3,D) and 'odd'}
    cache_by_condition = defaultdict(list)

    for images, dataset, condition, oddity_indices in tqdm(mochi_loader, desc='MOCHI: extract & cosine'):
        images = images.to(device)
        b_actual, n_images, c, h, w = images.shape

        images_flat = images.view(-1, c, h, w)
        feats_flat = model(images_flat)                   # (B*n_images, D)
        D = feats_flat.shape[-1]
        feats = feats_flat.view(b_actual, n_images, D)    # (B,n_images,D)

        # --- cosine baseline ---
        feats_norm = feats / feats.norm(dim=2, keepdim=True).clamp_min(1e-6)
        sim_mats = torch.bmm(feats_norm, feats_norm.transpose(1, 2))  # (B,n_images,n_images)

        diag = torch.eye(n_images, device=device).unsqueeze(0).expand(b_actual, -1, -1)
        sim_no_diag = sim_mats - diag
        sim_mean = sim_no_diag.sum(dim=2) / (n_images - 1)
        pred_oddity = torch.argmin(sim_mean, dim=1)

        correct = (pred_oddity == oddity_indices.to(device)).float()
        acc = adjust_accuracy(correct, 1 / n_images)

        total_correct_cos += acc.sum().item()
        total_samples += b_actual

        # per-condition tallies AND cache feats for SVM
        for i, (dataset_i, condition_i) in enumerate(zip(dataset, condition)):
            cond_name = f"{dataset_i}_{condition_i}"
            n_images_per_condition[cond_name] = n_images
            acc_per_condition_cos[cond_name] += acc[i].item()
            total_per_condition[cond_name] += 1

            # cache raw (unnormalized) features for SVM readout on CPU
            cache_by_condition[cond_name].append({
                "feats": feats[i].detach().cpu().float(),   # (n_images,D)
                "odd": int(oddity_indices[i]),
            })

    # reduce cosine baseline across ranks if needed
    if dist.is_initialized():
        t_correct = torch.tensor(total_correct_cos, device=device)
        t_samples = torch.tensor(total_samples, device=device)
        dist.all_reduce(t_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_samples, op=dist.ReduceOp.SUM)
        cosine_overall = (t_correct.item() / max(1, t_samples.item()))
        for cond_name in list(acc_per_condition_cos.keys()):
            cond_correct = torch.tensor(acc_per_condition_cos[cond_name], device=device)
            cond_total = torch.tensor(total_per_condition[cond_name], device=device)
            dist.all_reduce(cond_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(cond_total, op=dist.ReduceOp.SUM)
            acc_per_condition_cos[cond_name] = cond_correct.item() / max(1, cond_total.item())
    else:
        cosine_overall = total_correct_cos / max(1, total_samples)
        acc_per_condition_cos = {k: v / total_per_condition[k] for k, v in acc_per_condition_cos.items()}

    # --------------------------
    # linear SVM (reference implementation approach - trial by trial)
    # --------------------------
    np.random.seed(42)  # for reproducible random subsets
    
    # Collect all trials with their metadata
    all_trials = []
    for cond_name, trials in cache_by_condition.items():
        for trial in trials:
            trial['condition'] = cond_name
            all_trials.append(trial)
    
    # Store individual trial results for aggregation
    trial_results = []
    acc_per_condition_sd = defaultdict(list)
    
    # Process each individual trial (like reference implementation)
    for trial_idx, test_trial in enumerate(tqdm(all_trials, desc='MOCHI: same–different SVM')):
        test_feats = test_trial["feats"]  # (n_images, D)
        test_odd = test_trial["odd"]
        test_condition = test_trial["condition"]
        
        # Generate test pairs and labels for this trial
        X_test, y_test, test_pairs = _pair_diffs_and_labels(test_feats, test_odd)
        
        # Collect training data from all other trials in SAME condition
        train_trials = [trial for trial in all_trials 
                       if trial["condition"] == test_condition and trial is not test_trial]
        
        if not train_trials:  # Skip if no training data in same condition
            continue
            
        # Generate all training pairs from same condition
        X_train_all = []
        y_train_all = []
        
        for train_trial in train_trials:
            train_feats = train_trial["feats"]
            train_odd = train_trial["odd"]
            X_train_trial, y_train_trial, _ = _pair_diffs_and_labels(train_feats, train_odd)
            X_train_all.append(X_train_trial)
            y_train_all.append(y_train_trial)
        
        if not X_train_all:  # Skip if no training pairs
            continue
            
        X_train_full = torch.cat(X_train_all, dim=0).cpu().numpy()
        y_train_full = torch.cat(y_train_all, dim=0).cpu().numpy()
        
        # Multiple permutations with random subsets (like reference: n_permutations = 100)
        trial_accuracies = []
        
        for _ in range(svm_repeats):
            # Random subset of training data (like reference: subset_fraction = 0.75)
            n_train_samples = len(X_train_full)
            subset_size = max(1, int(train_fraction * n_train_samples))
            
            random_indices = np.random.permutation(n_train_samples)[:subset_size]
            X_train_subset = X_train_full[random_indices]
            y_train_subset = y_train_full[random_indices]
            
            # Train SVM (exactly like reference)
            clf = make_pipeline(StandardScaler(),
                              SVC(class_weight='balanced', probability=True))
            clf.fit(X_train_subset, y_train_subset)
            
            # Predict probabilities on test trial
            y_hat = clf.predict_proba(X_test.cpu().numpy())
            prob_different = y_hat[:, 0]  # P(different)
            
            # Compute average "different" probability for each image (like reference)
            n_images = test_feats.shape[0]
            diff_scores = []
            
            for img_idx in range(n_images):
                # Find pairs involving this image
                pair_probs = [prob_different[pair_idx] for pair_idx, (i, j) in enumerate(test_pairs)
                             if i == img_idx or j == img_idx]
                avg_diff_prob = np.mean(pair_probs) if pair_probs else 0.0
                diff_scores.append(avg_diff_prob)
            
            # Predict odd image (highest "different" probability)
            pred_odd = np.argmax(diff_scores)
            accuracy = 1.0 if pred_odd == test_odd else 0.0
            trial_accuracies.append(accuracy)
        
        # Average accuracy for this trial across all permutations
        trial_mean_acc = np.mean(trial_accuracies)
        trial_results.append(trial_mean_acc)
        acc_per_condition_sd[test_condition].append(trial_mean_acc)
    
    # Aggregate results by condition
    acc_per_condition_sd = {cond: np.mean(accs) for cond, accs in acc_per_condition_sd.items()}
    
    # Overall SVM accuracy
    samediff_overall = np.mean(trial_results) if trial_results else 0.0

    metrics = {
        'mochi_cosine_overall': cosine_overall,
        'mochi_SVM_overall': samediff_overall,
    }

    for cond, acc in acc_per_condition_cos.items():
        metrics[f'mochi_cosine/{cond}'] = acc
    for cond, acc in acc_per_condition_sd.items():
        metrics[f'mochi_samediff/{cond}'] = acc

    return metrics

