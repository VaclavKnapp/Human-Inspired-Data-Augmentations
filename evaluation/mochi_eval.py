import torch
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict

from utils.training_utils import adjust_accuracy
from sklearn.svm import LinearSVC

def _pair_diffs_and_labels(feats_3, odd_idx):
    """
    feats_3: (3, D) tensor (CPU or CUDA)
    odd_idx: int in {0,1,2}
    Returns:
      X: (3, D) torch.Tensor of pairwise diffs [p_same, p_neg1, p_neg2]
      y: (3,) torch.Tensor of labels [1, 0, 0]  (same=1, different=0)
      pairs: list of index pairs corresponding to rows of X
    """
    idxs = [0, 1, 2]
    same_pair = [i for i in idxs if i != odd_idx]  # the two non-odd images
    a, b = same_pair
    # order each pair (low, high) for a consistent sign convention
    a1, b1 = (a, b) if a < b else (b, a)
    X = []
    pairs = []

    def diff(i, j):
        i1, j1 = (i, j) if i < j else (j, i)
        return feats_3[i1] - feats_3[j1], (i1, j1)

    # 1 same, 2 different
    d_same, p_same = diff(a, b)
    X.append(d_same); pairs.append(p_same)
    for neg in [a, b]:
        d_neg, p_neg = diff(odd_idx, neg)
        X.append(d_neg); pairs.append(p_neg)

    X = torch.stack(X, dim=0)
    y = torch.tensor([1, 0, 0], dtype=torch.long, device=X.device)
    return X, y, pairs

@torch.no_grad()
def evaluate_mochi(
    model,
    mochi_loader,
    device,
    svm_repeats: int = 50,
    train_fraction: float = 0.75,
    svm_C: float = 1.0,
    svm_max_iter: int = 1000,
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
        assert n_images == 3, "MOCHI oddity trials should have exactly 3 images."

        images_flat = images.view(-1, c, h, w)
        feats_flat = model(images_flat)                   # (B*3, D)
        D = feats_flat.shape[-1]
        feats = feats_flat.view(b_actual, n_images, D)    # (B,3,D)

        # --- cosine baseline ---
        feats_norm = feats / feats.norm(dim=2, keepdim=True).clamp_min(1e-6)
        sim_mats = torch.bmm(feats_norm, feats_norm.transpose(1, 2))  # (B,3,3)

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
                "feats": feats[i].detach().cpu().float(),   # (3,D)
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
    # linear SVM (per condition, repeated resampling)
    # --------------------------
    rng = torch.Generator().manual_seed(0)  # torch RNG for indices; scikit-learn uses its own RNG
    sd_correct_sum = 0.0
    sd_total = 0
    acc_per_condition_sd = {}

    for cond_name, triplets in tqdm(cache_by_condition.items(), desc='MOCHI: same–different SVM'):
        n_trials = len(triplets)
        if n_trials < 4:
            # too few triplets to do proper resampling; fall back to chance-normalized 0
            # (or you could skip / warn)
            acc_per_condition_sd[cond_name] = 0.0
            sd_total += n_trials
            continue

        # per-condition stats
        cond_acc_sum = 0.0

        # iterate over target triplets (leave-one-out style; each with repeated resampling)
        for t_idx in range(n_trials):
            feats_t = triplets[t_idx]["feats"]  # (3,D)
            odd_t = triplets[t_idx]["odd"]

            # build three test pair diffs for triplet T
            X_test, _, pairs = _pair_diffs_and_labels(feats_t, odd_t)  # (3,D), labels not needed

            # collect correctness over repeats
            corrects = []

            for _ in range(svm_repeats):
                # sample training triplets from same condition, excluding T
                other_idxs = [i for i in range(n_trials) if i != t_idx]
                n_train = max(1, int(round(train_fraction * len(other_idxs))))
                # torch multinomial without replacement for reproducibility
                perm = torch.randperm(len(other_idxs), generator=rng).tolist()
                train_idxs = [other_idxs[i] for i in perm[:n_train]]

                # build training set (difference vectors + labels)
                X_train_list, y_train_list = [], []
                for i in train_idxs:
                    feats_i = triplets[i]["feats"]
                    odd_i = triplets[i]["odd"]
                    X_i, y_i, _ = _pair_diffs_and_labels(feats_i, odd_i)
                    X_train_list.append(X_i)
                    y_train_list.append(y_i)

                X_train = torch.cat(X_train_list, dim=0).cpu().numpy()
                y_train = torch.cat(y_train_list, dim=0).cpu().numpy()

                # fit linear SVM (same=1, different=0)
                clf = LinearSVC(C=svm_C, class_weight='balanced', max_iter=svm_max_iter)
                clf.fit(X_train, y_train)

                # test: score the 3 pairs, pick the most "same" pair (highest decision value)
                scores = clf.decision_function(X_test.cpu().numpy())  # shape (3,)
                best_pair_idx = int(scores.argmax())
                best_pair = pairs[best_pair_idx]  # (i, j) of the predicted "same" pair

                # infer predicted odd index = the index not in best_pair
                all_idx = {0, 1, 2}
                pred_odd = list(all_idx - set(best_pair))[0]
                corrects.append(1.0 if pred_odd == odd_t else 0.0)

            # mean correctness across repeats, then chance-normalize
            mean_correct = float(sum(corrects) / len(corrects))
            acc_t = adjust_accuracy(torch.tensor([mean_correct]), 1 / 3).item()
            cond_acc_sum += acc_t

        cond_acc = cond_acc_sum / n_trials
        acc_per_condition_sd[cond_name] = cond_acc
        sd_correct_sum += cond_acc * n_trials
        sd_total += n_trials

    samediff_overall = sd_correct_sum / max(1, sd_total)

    metrics = {
        'mochi_cosine_overall': cosine_overall,
        'mochi_SVM_overall': samediff_overall,
    }

    for cond, acc in acc_per_condition_cos.items():
        metrics[f'mochi_cosine/{cond}'] = acc
    for cond, acc in acc_per_condition_sd.items():
        metrics[f'mochi_samediff/{cond}'] = acc

    return metrics

