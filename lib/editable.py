from collections import namedtuple
from copy import copy
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.copy_and_replace import do_not_copy, copy_and_replace
from .utils.ingraph_update import IngraphGradientDescent


class BaseEditable(nn.Module):
    EditResult = namedtuple('EditResult', ['model', 'success', 'loss', 'complexity'])
    # model: a model that was adjusted out-of-place. Must be the same type as self (*Editable)
    # success: True if edit was successful, False otherwise
    # loss: objective function at the termination of the edit procedure
    # complexity: a measure of effort it took to edit model, e.g. number of SGD steps

    def edit(self, *data):
        # This should perform editing without changing current model and return EditResult
        return self.EditResult(self, success=False, loss=0.0, complexity=0.0)


class Editable(BaseEditable):

    def __init__(self, module: nn.Module, loss_function,
                 optimizer=IngraphGradientDescent(0.01), max_steps=float('inf'),
                 get_editable_parameters=lambda module: module.parameters(),
                 is_edit_finished=lambda loss, **kwargs: loss.item() <= 0,
                 ):
        """
        Editable module that attempts to change model by performing SGD (with optional momentum and rms scaling)
        :param module: a torch module that will be edited
        :param loss_function: objective function(model(inputs), targets) that is minimized by editor.
            By default this function should be non-negative and loss == 0 is a trigger to finish editing
        :param optimizer: in-graph optimizer that creates updated copies of model
        :param get_editable_parameters: a function(Editable.module) that takes the wrapped module and returns
            an iterable of parameters that should affected by edits, defaults to all parameters inside Editable.module
        :param is_edit_finished: a function(loss, prediction, **local variables) that returns True if edit is finished
        """
        super().__init__()
        self.module, self.loss_function, self.optimizer = module, loss_function, optimizer
        self.get_editable_parameters = get_editable_parameters
        self.is_edit_finished = is_edit_finished
        self.max_steps = max_steps

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def edit(self, inputs, targets, max_steps=None, model_kwargs=None, loss_kwargs=None, opt_kwargs=None, **kwargs):
        """
        Attempts to edit model (out-of-place) and return an edited copy
        :param inputs: data that is fed into the model
        :param targets: reference answers that are fed into loss function
        :param max_steps: after this many gradient steps the process is terminated
        :param model_kwargs: optional extra model inputs, used as model(inputs, **model_params)
        :param loss_kwargs: optional extra loss parameters, self.loss_function(model(inputs), targets, **loss_params)
        :param opt_kwargs: optional overrides for optimizer.get_initial_state
        :param kwargs: extra parameters passed to optimizer.step
        :returns: edited_model, is_edit_successful, final_loss, gradients_steps
        :rtype: Editable.EditResult
        """
        model_kwargs, loss_kwargs, opt_kwargs = model_kwargs or {}, loss_kwargs or {}, opt_kwargs or {}
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)
        editable = self

        for step in count():
            prediction = editable(inputs, **model_kwargs)
            loss = self.loss_function(prediction, targets, **loss_kwargs)

            if self.is_edit_finished(**locals()):
                return self.EditResult(editable, success=True, loss=loss, complexity=step)
            elif step >= (max_steps or self.max_steps):
                return self.EditResult(editable, success=False, loss=loss, complexity=step)

            optimizer_state, editable = self.optimizer.step(
                optimizer_state, editable, loss, parameters=editable.get_editable_parameters(editable.module), **kwargs)

    def extra_repr(self):
        return "max_steps={}, loss_function={}".format(self.max_steps, repr(self.loss_function))


class SequentialWithEditable(BaseEditable):
    def __init__(self, *args):
        """ A chain of modules with exactly one Editable, edit procedure will only compute pre-editable modules once """
        super().__init__()
        pre_editable, editable, post_editable = [], None, []
        for module in args:
            if isinstance(module, BaseEditable):
                assert editable is None, "SequentialEditable only supports one Editable module for now"
                editable = module
            elif editable is None:
                pre_editable.append(module)
            else:
                post_editable.append(module)

        assert editable is not None, "SequentialEditable must have one Editable at init, got 0"
        self.prefix_layers = nn.Sequential(*pre_editable)
        self.editable = editable if len(post_editable) == 0 else self._editable_with_suffix(editable, *post_editable)

    def forward(self, *args, **kwargs):
        return self.editable(self.prefix_layers(*args, **kwargs))

    def edit(self, inputs, *args, **kwargs):
        result = self.editable.edit(self.prefix_layers(inputs), *args, **kwargs)
        with do_not_copy(self.prefix_layers, *self.prefix_layers.parameters(), *self.prefix_layers.buffers()):
            edited_model = copy_and_replace(self, replace={self.editable: result.model})
        return self.EditResult(edited_model, *result[1:])

    @staticmethod
    def _editable_with_suffix(base_editable: Editable, *suffix):
        new_editable = copy(base_editable)
        new_editable.module = nn.Sequential(base_editable.module, *suffix)
        new_editable.get_editable_parameters = lambda module: base_editable.get_editable_parameters(module[0])
        return new_editable


class RehearsalEditable(Editable):
    def __init__(self, *args, rehearsal_loss_weight=1.0, get_rehearsals, **kwargs):
        super().__init__(*args, **kwargs)
        self.rehearsal_loss_weight = rehearsal_loss_weight
        self.get_rehearsals = get_rehearsals

    def edit(self, inputs, targets, max_steps=None, model_kwargs=None, loss_kwargs=None, opt_kwargs={}, **kwargs):
        model_kwargs, loss_kwargs, opt_kwargs = model_kwargs or {}, loss_kwargs or {}, opt_kwargs or {}
        optimizer_state = self.optimizer.get_initial_state(self, **opt_kwargs)
        editable = self

        X_batch = self.get_rehearsals(inputs)
        vanilla_probs = F.softmax(editable(X_batch, **model_kwargs), dim=-1).detach()   
        for step in count():
            prediction = editable(inputs, **model_kwargs)
            
            loss = self.loss_function(prediction, targets, **loss_kwargs)
            
            if self.is_edit_finished(**locals()):
                return self.EditResult(editable, success=True, loss=loss, complexity=step)
            elif step >= (max_steps or self.max_steps):
                return self.EditResult(editable, success=False, loss=loss, complexity=step)

            current_logp = F.log_softmax(editable(X_batch, **model_kwargs), dim=-1)
            batch_loss = F.kl_div(current_logp, vanilla_probs)
            total_loss = loss + self.rehearsal_loss_weight * batch_loss

            with do_not_copy(self.get_rehearsals):
                optimizer_state, editable = self.optimizer.step(
                    optimizer_state, editable, total_loss, parameters=editable.get_editable_parameters(editable.module), **kwargs)
