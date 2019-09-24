import torch
from .basic import to_one_hot


def gumbel_noise(*sizes, epsilon=1e-9, **kwargs):
    """ Sample noise from gumbel distribution """
    return -torch.log(-torch.log(torch.rand(*sizes, **kwargs) + epsilon) + epsilon)


def gumbel_softmax(logits, dim=-1, tau=1.0, noise=1.0, hard=False, **kwargs):
    """
    Softmax with gumbel noise
    :param logits: inputs for softmax
    :param dim: normalize softmax along this dimension
    :param tau: gumbel softmax temperature
    :param hard: if True, works like onehot(sample) during forward pass,
        gumbel-softmax for backward pass
    :return: gumbel-softmax "probabilities", tensor of same shape as logits
    """
    if noise != 0:
        z = gumbel_noise(*logits.shape, device=logits.device, dtype=logits.dtype)
        logits = logits + noise * z
    if tau != 1.0:
        logits = logits / tau

    probs_gumbel = torch.softmax(logits, dim=dim)

    if hard:
        _, argmax_indices = torch.max(probs_gumbel, dim=dim)
        hard_argmax_onehot = to_one_hot(argmax_indices, depth=logits.shape[dim])
        if dim != -1 and dim != len(logits.shape) - 1:
            new_dim_order = list(range(len(logits.shape) - 1))
            new_dim_order.insert(dim, -1)
            hard_argmax_onehot = hard_argmax_onehot.permute(*new_dim_order)

        # forward pass: onehot sample, backward pass: gumbel softmax
        probs_gumbel = (hard_argmax_onehot - probs_gumbel).detach() + probs_gumbel

    return probs_gumbel


def gumbel_sigmoid(logits, tau=1.0, noise=1.0, hard=False, **kwargs):
    """
    A special case of gumbel softmax with 2 classes: [logit] and 0
    :param logits: sigmoid inputs
    :param tau: same as gumbel softmax temperature
    :param hard: if True, works like bernoulli sample for forward pass,
        gumbel sigmoid for backward pass
    :return: tensor with same shape as logits
    """
    if noise != 0.0:
        z1 = gumbel_noise(*logits.shape, device=logits.device, dtype=logits.dtype)
        z2 = gumbel_noise(*logits.shape, device=logits.device, dtype=logits.dtype)
        logits = logits + noise *(z1 - z2)
    if tau != 1.0:
        logits /= tau
    sigm = torch.sigmoid(logits)
    if hard:
        hard_sample = torch.ge(sigm, 0.5).to(dtype=logits.dtype)
        sigm = (hard_sample - sigm).detach() + sigm
    return sigm
