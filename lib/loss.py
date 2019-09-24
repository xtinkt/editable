"""
    Loss functions
"""

import torch
from torch import nn as nn
import torch.nn.functional as F


def kl_distill_loss(logits, ref_probs):
    """
    kullback leibler divergence
    """
    return F.kl_div(F.log_softmax(logits, dim=-1), ref_probs)


def contrastive_cross_entropy(logits, target, margin=0.0):
    """
    A special loss that is similar to crossentropy but becomes exactly zero if
    logp(target) >= max(logp(all_excluding_target)) + margin
    Used for classification edits
    """
    logp = F.log_softmax(logits, dim=-1)
    target_one_hot = F.one_hot(target, num_classes=logp.shape[-1])
    logp_target = (logp * target_one_hot.to(logits.dtype)).sum(-1)
    logp_others = torch.where(target_one_hot.to(torch.uint8), torch.full_like(logp, -float('inf')), logp)
    return F.relu(margin + logp_others.max(dim=-1)[0] - logp_target).mean()


def threshold_mse(predictions, targets, threshold=0.0, reduction_axes=None):
    """
    Like mean squared error but becomes exactly zero if
    sum of squared errors along reduction axes is below threshold
    used for regression edits
    """
    squared_error = (predictions - targets) ** 2
    if reduction_axes is not None:
        squared_error = squared_error.sum(reduction_axes)
    return F.relu(squared_error - threshold).mean()
