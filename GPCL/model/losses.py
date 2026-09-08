"""Confidence-weighted pseudo-label-guided InfoNCE, paper Eq. (11)."""

import torch


def confidence_weighted_infonce(
    z: torch.Tensor,
    classes: torch.Tensor,
    confidence: torch.Tensor,
    temperature: float,
    negatives_per_anchor: int,
) -> torch.Tensor:
    similarities = z @ z.T / temperature
    losses = []
    batch = len(z)
    all_indices = torch.arange(batch, device=z.device)
    for i in range(batch):
        positive_candidates = all_indices[(classes == classes[i]) & (all_indices != i)]
        negative_candidates = all_indices[classes != classes[i]]
        if len(positive_candidates) == 0 or len(negative_candidates) == 0:
            continue
        p = positive_candidates[torch.randint(len(positive_candidates), (1,), device=z.device)].item()
        take = negatives_per_anchor
        n = negative_candidates[torch.randint(len(negative_candidates), (take,), device=z.device)]
        pair_weight = ((confidence[i] + confidence[p]) * 0.5).clamp_min(1e-8)
        positive_logit = similarities[i, p] + pair_weight.log()
        denominator = torch.logsumexp(torch.cat((positive_logit[None], similarities[i, n])), dim=0)
        losses.append(-positive_logit + denominator)
    if not losses:
        # Keep a valid autograd connection in the extremely unlikely one-class batch.
        return z.sum() * 0.0
    return torch.stack(losses).mean()

