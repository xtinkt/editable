from collections import OrderedDict
from torch.optim.optimizer import Optimizer
import numpy as np


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            return param_group['lr']
    raise ValueError("Could not infer learning rate from optimizer {}".format(optimizer))


class OneCycleSchedule(Optimizer):
    """ A simplified torch lr schedule that updates learning rate before every opt.step """

    def __init__(self, optimizer, **kwargs):
        """
        :type optimizer: torch.optim.Optimizer
        :param kwargs: see self.update_learning_rate
        """
        self.learning_rate_opts = kwargs
        self.opt = optimizer
        self.step_count = 0

    def step(self, **kwargs):
        self.current_lr = self.update_learning_rate(t=self.step_count, **self.learning_rate_opts)
        res = self.opt.step(**kwargs)
        self.step_count += 1
        return res

    def state_dict(self, **kwargs):
        return OrderedDict([
            ('optimizer_state_dict', self.opt.state_dict(**kwargs)),
            ('learning_rate_opts', self.learning_rate_opts),
            ('step_count', self.step_count)
        ])

    def load_state_dict(self, state_dict, load_step=True, load_opts=True, **kwargs):
        self.learning_rate_opts = state_dict['learning_rate_opts'] if load_opts else self.learning_rate_opts
        self.step_count = state_dict['step_count'] if load_step else self.step_count
        return self.opt.load_state_dict(state_dict['optimizer_state_dict'], **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.opt, attr)

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=10000,
                             decay_rate=0.2, learning_rate_min=1e-5):
        """ Learning rate with linear warmup and exponential decay """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        return lr
