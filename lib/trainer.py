import torch
import numpy as np
from torch.nn import functional as F

from .evaluate import classification_error, evaluate_quality
from .utils import training_mode, BaseTrainer
from .editable import Editable
from .loss import kl_distill_loss


class EditableTrainer(BaseTrainer):
    def __init__(self, model: Editable, loss_function, error_function=classification_error, opt=None,
                 stability_coeff=0.01, editability_coeff=0.01, max_norm=None, **kwargs):
        """ A simple optimizer that trains to minimize classification or regression loss """
        opt = opt if opt is not None else torch.optim.Adam(model.parameters())
        super().__init__(model, loss_function=loss_function, opt=opt, error_function=error_function, **kwargs)
        self.stability_coeff, self.editability_coeff, self.max_norm = stability_coeff, editability_coeff, max_norm

    def train_on_batch(self, x_batch, y_batch, x_edit, y_edit, prefix='train/', is_train=True, **kwargs):
        """ Performs a single gradient update and reports metrics """
        x_batch, y_batch = map(torch.as_tensor, (x_batch, y_batch))
        self.opt.zero_grad()

        with training_mode(self.model, is_train=is_train):
            logits = self.model(x_batch)

        main_loss = self.loss_function(logits, y_batch).mean()

        with training_mode(self.model, is_train=False):
            model_edited, success, editability_loss, complexity = self.model.edit(x_edit, y_edit, **kwargs)
            logits_updated = model_edited(x_batch)

        stability_loss = - (F.softmax(logits.detach(), dim=1) * F.log_softmax(logits_updated, dim=1)).sum(dim=1).mean()

        final_loss = main_loss + self.stability_coeff * stability_loss + self.editability_coeff * editability_loss

        metrics = dict(
            final_loss=final_loss.item(), stability_loss=stability_loss.item(),
            editability_loss=editability_loss.item(), main_loss=main_loss.item(),
        )

        final_loss.backward()

        if self.max_norm is not None:
            metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        self.opt.step()

        return self.record(**metrics, prefix=prefix)

    def evaluate_metrics(self, X, y, X_edit=None, y_edit=None, prefix='val/', **kwargs):
        """
        For each sample in X_edit, y_edit attempts to train model and evaluates trained model quality
        :param X: data for quality evaluaton
        :param y: targets for quality evaluaton
        :param X_edit: sequence of data for training model on
        :param y_edit: sequence of targets for training model on
        :param prefix: tensorboard metrics will be written under this prefix
        :param kwargs: extra parameters for error function
        :return: dictionary of metrics
        """
        assert (X_edit is None) == (y_edit is None), "provide either both X_edit and y_edit or none of them"
        if X_edit is None:
            num_classes = y.max() + 1
            ind = np.random.permutation(len(X))[:10]
            X_edit = X[ind]
            y_edit = (y[ind] + torch.randint_like(y[ind], 1, num_classes)) % num_classes

        return self.record(**evaluate_quality(
            self.model, X, y, X_edit, y_edit, error_function=self.error_function, **kwargs), prefix=prefix)

    def extra_repr(self):
        line = "stability_coeff = {}, editability_coeff = {}, max_norm = {}".format(
            self.stability_coeff, self.editability_coeff, self.max_norm)
        line += '\nloss = {} '.format(self.loss_function)
        line += '\nopt = {} '.format(self.opt)
        return line

    
class DistillationEditableTrainer(EditableTrainer):
    def __init__(self, model, **kwargs):
        return super().__init__(model, loss_function=kl_distill_loss, **kwargs)
    
    def train_on_batch(self, x_batch, logits_batch, *args, **kwargs):
        return super().train_on_batch(x_batch, logits_batch, *args, is_train=True, **kwargs)
