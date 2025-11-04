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


def generate_delta_vectors(trials, features, oddities, images):
    """
    Generate delta vectors exactly like reference implementation
    """
    def delta(a, b, xtype='diff'):
        if xtype == 'diff':
            x = a - b
        elif xtype == 'abs':
            x = np.array(np.abs(a - b))
        elif xtype == 'sqrt':
            x = np.sqrt(np.array(np.abs(a - b)))
        elif xtype == 'sqr':
            x = (np.array(np.abs(a - b)))**2
        elif xtype == 'product':
            x = a * b
        return x

    trial_markers = []
    deltas = []
    labels = []

    for i_trial in trials:
        # Get indices for this trial
        trial_indices = [images[key] for key in images.keys() if key.split('-')[0] == i_trial]
        i_features = features[trial_indices]
        oddity_index = oddities[i_trial]

        pairs = []  # Store pairs for this trial

        for i in range(len(i_features)):
            for j in range(i+1, len(i_features)):
                i_delta = delta(i_features[i], i_features[j])
                deltas.append(i_delta)
                trial_markers.append(i_trial)
                pairs.append((i, j))

                # Label: 0 if one is odd (different), 1 if both are same
                if (i == oddity_index) or (j == oddity_index):
                    labels.append(0)  # different
                else:
                    labels.append(1)  # same

    # Create location mapping for the last trial (used for prediction)
    all_inds = np.array(range(len(i_features)))
    pairs_array = np.array(pairs)
    location_of_indices = {}
    for i in all_inds:
        # Find pairs involving image i
        mask = (pairs_array == i).any(axis=1)
        location_of_indices[i] = np.where(mask)[0]

    info = {
        'deltas': np.array(deltas),
        'labels': np.array(labels),
        'trials': trial_markers,
        'inds': all_inds,
        'locs': location_of_indices
    }

    return info

@torch.no_grad()
def evaluate_mochi(
    model,
    mochi_loader,
    device,
    svm_repeats: int = 100,  # n_permutations like reference
    train_fraction: float = 0.75,  # subset_fraction in reference
    svm_C: float = 1.0,
    svm_max_iter: int = 5000,
):
    """
    Computes BOTH:
      1) cosine-argmin oddity (zero-shot)
      2) same–different linear SVM readout (GLOBAL cross-condition training like reference)

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
    # linear SVM (reference implementation approach - CONDITION-SPECIFIC TRAINING)
    # --------------------------
    np.random.seed(42)  # for reproducible random subsets

    # Store individual trial results for aggregation
    trial_results = []
    acc_per_condition_sd = defaultdict(list)

    print("Processing each condition for SVM evaluation...")
    # iterate through each condition (exactly like notebook)
    for cond_name, trials_in_condition in tqdm(cache_by_condition.items(), desc='MOCHI: same–different SVM'):

        # Convert cached features to reference format for this condition only
        features = []
        oddities = {}
        images = {}
        feature_idx = 0

        for trial_idx, trial in enumerate(trials_in_condition):
            trial_id = f"{trial_idx}"  # Simple trial ID within condition
            oddities[trial_id] = trial["odd"]
            trial_feats = trial["feats"].numpy()  # (n_images, D)

            # Store features and create image mapping
            for img_idx in range(trial_feats.shape[0]):
                features.append(trial_feats[img_idx].flatten())
                images[f'{trial_id}-{img_idx}'] = feature_idx
                feature_idx += 1

        features = np.array(features)
        trials = list(oddities.keys())

        # Generate delta vectors WITHIN this condition only
        diffs = generate_delta_vectors(trials, features, oddities, images)

        # Process each trial in this condition
        for trial_idx, trial in enumerate(trials_in_condition):
            i_trial = str(trial_idx)
            i_oddity_index = trial["odd"]

            # Extract indices and labels for training SVM (OTHER TRIALS IN SAME CONDITION)
            train_indices = [i for i, trial_name in enumerate(diffs['trials']) if trial_name != i_trial]

            if not train_indices:  # Skip if no training data
                continue

            # vectors for the difference between each image vector
            X_train = diffs['deltas'][train_indices, :]
            # labels for whether each vector was 'same' or 'different'
            y_train = diffs['labels'][train_indices]

            # Extract indices and labels for testing SVM
            test_indices = [i for i, trial_name in enumerate(diffs['trials']) if trial_name == i_trial]

            if not test_indices:  # Skip if no test data
                continue

            # vectors for the difference between each image vector
            X_test = diffs['deltas'][test_indices, :]

            # for each iteration use X% of the available trials
            len_subset = int(train_fraction * len(X_train))

            # prep for each iteration
            choices = []

            for _ in range(svm_repeats):  # n_permutations = 100 like reference
                # identify random subset of delta vectors to train on
                random_subset = np.random.permutation(len(X_train))[:len_subset]

                # define model to train a linear readout
                clf = make_pipeline(StandardScaler(),
                                  SVC(class_weight='balanced', probability=True))
                # fit training data
                clf.fit(X_train[random_subset, :], y_train[random_subset])
                # predict performance on this trial
                y_hat = clf.predict_proba(X_test)

                # identify which image has the highest probability of being different
                prob_different = y_hat[:, 0] if y_hat.shape[1] > 1 else 1 - y_hat[:, 0]  # P(different)

                # Calculate per-image difference scores exactly like notebook
                n_images = trial["feats"].shape[0]
                i_diffs = []
                for i in diffs['inds']:
                    # Get pairs involving this image for the current trial
                    pair_probs = [prob_different[pair_idx] for pair_idx in diffs['locs'][i]
                                 if pair_idx < len(prob_different)]
                    avg_diff_prob = np.mean(pair_probs) if pair_probs else 0.0
                    i_diffs.append(avg_diff_prob)

                # determine whether the model-selected oddity matches ground truth
                i_trial_accuracy = i_oddity_index == np.argmax(i_diffs)
                # save
                choices.append(i_trial_accuracy)

            # Average accuracy for this trial across all permutations
            trial_mean_acc = np.mean(choices)
            trial_results.append(trial_mean_acc)
            acc_per_condition_sd[cond_name].append(trial_mean_acc)

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


@torch.no_grad()
def evaluate_mochi_siamese(model, mochi_loader, device):
    """
    Evaluate Siamese network on Mochi benchmark.

    Strategy:
    For each trial with n_images where one is the odd one out:
    1. Compute pairwise similarity scores for all pairs
    2. For each image, compute its average similarity with all other images
    3. The image with the lowest average similarity is predicted as the odd one out

    Returns:
        dict: Metrics including overall accuracy and per-condition accuracy
    """
    model.eval()

    total_correct = 0.0
    total_samples = 0
    acc_per_condition = defaultdict(float)
    total_per_condition = defaultdict(int)

    for images, dataset, condition, oddity_indices in tqdm(mochi_loader, desc='MOCHI [Siamese]'):
        images = images.to(device)
        b_actual, n_images, c, h, w = images.shape

        # Compute all pairwise similarities for each batch item
        avg_similarities = []

        for batch_idx in range(b_actual):
            # Compute average similarity for each image with all other images
            img_avg_sims = []

            for i in range(n_images):
                # Get similarity of image i with all other images
                sims_with_others = []
                for j in range(n_images):
                    if i != j:
                        img_i = images[batch_idx, i:i+1]  # (1, c, h, w)
                        img_j = images[batch_idx, j:j+1]  # (1, c, h, w)

                        # Compute similarity (returns logits, apply sigmoid)
                        logits = model(img_i, img_j)
                        sim = torch.sigmoid(logits).item()
                        sims_with_others.append(sim)

                # Average similarity for this image
                avg_sim = sum(sims_with_others) / len(sims_with_others)
                img_avg_sims.append(avg_sim)

            avg_similarities.append(img_avg_sims)

        avg_similarities = torch.tensor(avg_similarities, device=device)  # (b_actual, n_images)

        # The odd one out has the LOWEST average similarity
        pred_oddity = torch.argmin(avg_similarities, dim=1)

        # Calculate accuracy
        correct = (pred_oddity == oddity_indices.to(device)).float()
        acc = adjust_accuracy(correct, 1.0 / n_images)

        total_correct += acc.sum().item()
        total_samples += b_actual

        # Per-condition tallies
        for i, (dataset_i, condition_i) in enumerate(zip(dataset, condition)):
            cond_name = f"{dataset_i}_{condition_i}"
            acc_per_condition[cond_name] += acc[i].item()
            total_per_condition[cond_name] += 1

    # Reduce across ranks if needed
    if dist.is_initialized():
        t_correct = torch.tensor(total_correct, device=device)
        t_samples = torch.tensor(total_samples, device=device)
        dist.all_reduce(t_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_samples, op=dist.ReduceOp.SUM)
        overall_acc = (t_correct.item() / max(1, t_samples.item()))

        for cond_name in list(acc_per_condition.keys()):
            cond_correct = torch.tensor(acc_per_condition[cond_name], device=device)
            cond_total = torch.tensor(total_per_condition[cond_name], device=device)
            dist.all_reduce(cond_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(cond_total, op=dist.ReduceOp.SUM)
            acc_per_condition[cond_name] = cond_correct.item() / max(1, cond_total.item())
    else:
        overall_acc = total_correct / max(1, total_samples)
        acc_per_condition = {k: v / total_per_condition[k] for k, v in acc_per_condition.items()}

    metrics = {
        'mochi_siamese_overall': overall_acc,
    }

    for cond, acc in acc_per_condition.items():
        metrics[f'mochi_siamese/{cond}'] = acc

    return metrics