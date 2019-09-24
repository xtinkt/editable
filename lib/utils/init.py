import torch
import torch.nn as nn
from torch import nn as nn
from torch.jit import ScriptModule
from torch.nn import functional as F


class ModuleWithInit(nn.Module):
    """ Base class for pytorch module with data-aware initializer on first batch """
    def __init__(self):
        super().__init__()
        assert not hasattr(self, '_is_initialized_bool')
        assert not hasattr(self, '_is_initialized_tensor')
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized_* so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules

    def initialize(self, *args, **kwargs):
        """ initialize module tensors using first batch of data """
        raise NotImplementedError("Please implement ")

    def is_initialized(self):
        """ whether data aware initialization was already performed """
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        return self._is_initialized_bool

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class ScriptModuleWithInit(ModuleWithInit, ScriptModule):
    """ Base class for pytorch module with data-aware initializer on first batch """
    def __init__(self, optimize=True, **kwargs):
        ScriptModule.__init__(self, optimize=optimize, **kwargs)
        ModuleWithInit.__init__(self)


def init_normalized_(x, init_=nn.init.normal_, dim=-1, **kwargs):
    """ initialize x inp-place by sampling random normal values and normalizing them over dim """
    init_(x)
    x.data = F.normalize(x, dim=dim, **kwargs)
    return x
