import contextlib
import gc
import os
import time
from collections import Counter
from itertools import chain

import torch
from torch import nn as nn


def to_one_hot(y, depth=None):
    r"""
    Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.
    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.
    Args:
        y: input integer (IntTensor, LongTensor or Variable) of any shape
        depth (int):  the size of the one hot dimension
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot


def dot(x, y):
    """ numpy-like dot product """
    out_flat = x.view(-1, x.shape[-1]) @ y.view(y.shape[0], -1)
    return out_flat.view(*x.shape[:-1], *y.shape[1:])


def batch_outer_sum(*tensors):
    """
    :param tensors: each matrix should have shape [..., d_i]
    :returns: [..., d_0, d_1, ..., d_N] where N = len(tensors)
        output[..., i, j, k] = tensors[0][..., i] + tensors[1][..., j] + tensors[2][..., k]
    """
    outer_sum = None
    for i, tensor in enumerate(tensors):
        broadcaster = [None] * len(tensors)
        broadcaster[i] = slice(tensor.shape[-1])
        broadcaster = tuple([...] + broadcaster)
        outer_sum = tensor[broadcaster] if i == 0 else outer_sum + tensor[broadcaster]
    return outer_sum


def batch_outer_product(*tensors):
    """
    :param tensors: each matrix should have shape [..., d_i]
    :returns: [..., d_0, d_1, ..., d_N] where N = len(tensors)
        output[..., i, j, k] = tensors[0][..., i] * tensors[1][..., j] * tensors[2][..., k]
    """
    prefix_shape = tensors[0].shape[:-1]
    assert len(tensors) + len(prefix_shape) <= ord('z') - ord('a')

    prefix_chars = ''.join(map(chr, range(ord('a'), ord('a') + len(prefix_shape))))
    dim_chars = ''.join(map(chr, range(ord('a') + len(prefix_shape), ord('a') + len(prefix_shape) + len(tensors))))
    einsum_lhs = ','.join(prefix_chars + d_i for d_i in dim_chars)
    einsum_rhs = prefix_chars + dim_chars
    return torch.einsum("{}->{}".format(einsum_lhs, einsum_rhs), *tensors)


def straight_through_grad(function, **kwargs):
    """
    modify function so that it is applied normally but excluded from backward pass
    :param function: callable(*inputs) -> *outputs, number and shape of outputs must match that of inputs,
    :param kwargs: keyword arguments that will be sent to each function call
    """
    def f_straight_through(*inputs):
        outputs = function(*inputs, **kwargs)
        single_output = isinstance(outputs, torch.Tensor)
        if single_output:
            outputs = [outputs]

        assert isinstance(outputs, (list, tuple)) and len(outputs) == len(inputs)
        outputs = type(outputs)(
            input + (output - input).detach()
            for input, output in zip(inputs, outputs)
        )
        return outputs[0] if single_output else outputs

    return f_straight_through


def nop(x):
    return x


@contextlib.contextmanager
def nop_ctx():
    yield None


class Nop(nn.Module):
    def forward(self, x):
        return x


class Residual(nn.Sequential):
    def forward(self, x):
        return super().forward(x) + x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)


@contextlib.contextmanager
def training_mode(*modules, is_train:bool):
    group = nn.ModuleList(modules)
    was_training = {module: module.training for module in group.modules()}
    try:
        yield group.train(is_train)
    finally:
        for key, module in group.named_modules():
            if module in was_training:
                module.training = was_training[module]
            else:
                raise ValueError("Model was modified inside training_mode(...) context, could not find {}".format(key))


def free_memory(sleep_time=0.1):
    """ Black magic function to free torch memory and some jupyter whims """
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)


def infer_model_device(model: nn.Module):
    """ infers model device as the device where the majority of parameters and buffers are stored """
    device_stats = Counter(
        tensor.device for tensor in chain(model.parameters(), model.buffers())
        if torch.is_tensor(tensor)
    )
    return max(device_stats, key=device_stats.get)


class Lambda(nn.Module):
    def __init__(self, func):
        """ :param func: call this function during forward """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if run_from_ipython():
    from IPython.display import clear_output
else:
    def clear_output(*args, **kwargs):
        os.system('clear')

        
class OptimizerList(torch.optim.Optimizer):
    def __init__(self, *optimizers):
        self.optimizers = optimizers
    
    def step(self):
        return [opt.step() for opt in self.optimizers]
    
    def zero_grad(self):
        return [opt.zero_grad() for opt in self.optimizers]
    
    def add_param_group(self, *args, **kwargs):
        raise ValueError("Please call add_param_group in one of self.optimizers")
        
    def __getstate__(self):
        return [opt.__getstate__() for opt in self.optimizers]
    
    def __setstate__(self, state):
        return [opt.__setstate__(opt_state) for opt, opt_state in zip(self.optimizers, state)]
    
    def __repr__(self):
        return repr(self.optimizers)
    
    def state_dict(self, **kwargs):
        return {"opt_{}".format(i): opt.state_dict(**kwargs) for i, opt in enumerate(self.optimizers)}
    
    def load_state_dict(self, state_dict, **kwargs):
        return [
            opt.load_state_dict(state_dict["opt_{}".format(i)])
            for i, opt in enumerate(self.optimizers)
        ]
