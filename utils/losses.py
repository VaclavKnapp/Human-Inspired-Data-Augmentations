import torch
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_p = 1 - F.cosine_similarity(anchor, positive)
        dist_n = 1 - F.cosine_similarity(anchor, negative)

        loss = torch.clamp(dist_p - dist_n + self.margin, min=0)
        return loss.mean()

class HingeLoss(torch.nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, dist_p, dist_n):
        loss = torch.clamp(dist_p - dist_n + self.margin, min=0)
        return loss.mean()


class SingleTripletMultiSimilarityLoss(torch.nn.Module):
    # Simplified implementation of https://arxiv.org/abs/1904.06627
    def __init__(self, alpha=2, beta=50, base=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, anchor, positive, negative):
        s_ap = F.cosine_similarity(anchor, positive, dim=-1)
        s_an = F.cosine_similarity(anchor, negative, dim=-1)
        s_pn = F.cosine_similarity(positive, negative, dim=-1)

        pos_loss = (1.0 / self.alpha) * torch.log(1 + torch.exp(-self.alpha * (s_ap - self.base)))

        neg_loss = (1.0 / self.beta) * torch.log(
            1 + torch.exp(self.beta * (s_an - self.base)) +
            torch.exp(self.beta * (s_pn - self.base))
        )

        return pos_loss + neg_loss

class Oddity_Loss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, anchor_emb, positive_emb, negative_emb):
        anchor_emb = F.normalize(anchor_emb, p=2, dim=-1)
        positive_emb = F.normalize(positive_emb, p=2, dim=-1)
        negative_emb = F.normalize(negative_emb, p=2, dim=-1)

        sim_ap = torch.einsum('nd,nd->n', anchor_emb, positive_emb)
        sim_an = torch.einsum('nd,nd->n', anchor_emb, negative_emb)
        sim_pn = torch.einsum('nd,nd->n', positive_emb, negative_emb)

        mean_sim_anchor = (sim_ap + sim_an) / 2
        mean_sim_positive = (sim_ap + sim_pn) / 2
        mean_sim_negative = (sim_pn + sim_an) / 2

        logits = -torch.stack([mean_sim_anchor, mean_sim_positive, mean_sim_negative], dim=1)
        logits /= self.temperature

        target = torch.full((anchor_emb.size(0),), 2, dtype=torch.long, device=anchor_emb.device)
        loss = self.loss_fn(logits, target)
        return loss


class ContrastiveSimilarityLoss(torch.nn.Module):
    """
    Standard pairwise contrastive loss for metric learning.
    Given matched and unmatched pairs, learns a representation space
    where semantically similar inputs are closer together.

    This loss operates on pairs (img1, img2) with binary labels:
    - label=1.0 for similar/matched pairs (should be close)
    - label=0.0 for dissimilar/unmatched pairs (should be far)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b, label):
        """
        Args:
            emb_a: embeddings of first input  (N, D)
            emb_b: embeddings of second input (N, D)
            label: 1.0 if same category, 0.0 if different (N,)

        Returns:
            Binary cross-entropy loss
        """
        # Normalize embeddings to unit sphere (makes dot product = cosine similarity)
        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        # Compute cosine similarity via dot product
        similarity = (emb_a * emb_b).sum(dim=-1)  # (N,)

        # Scale by temperature (higher temp = softer decisions)
        logit = similarity / self.temperature

        # Binary cross-entropy: pushes similar pairs to high similarity, dissimilar to low
        return F.binary_cross_entropy_with_logits(logit, label)