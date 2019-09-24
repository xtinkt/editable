import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from .utils import infer_model_device, training_mode, process_in_chunks, check_numpy, nop
from .editable import Editable


def classification_error(model: nn.Module, X_test, y_test, batch_size=1024, device=None):
    device = device or infer_model_device(model)
    with torch.no_grad(), training_mode(model, is_train=False):
        val_logits = process_in_chunks(
            model, torch.as_tensor(X_test, device=device), batch_size=batch_size)
        val_logits = check_numpy(val_logits)
        error_rate = (check_numpy(y_test) != np.argmax(val_logits, axis=1)).mean()
    return error_rate

def eval_func(model: nn.Module, X_test, batch_size=1024, device='cuda'):
    with torch.no_grad(), training_mode(model, is_train=False):
        val_logits = process_in_chunks(
            model, torch.as_tensor(X_test, device=device), batch_size=batch_size)
        val_logits = check_numpy(val_logits)
    return val_logits

def calculate_edit_statistics(editable_model: Editable, X_test, y_test, X_edit, y_edit,
                              error_function=classification_error, progressbar=None, **kwargs):
    """
    For each sample in X_edit, y_edit attempts to train model and evaluates trained model quality
    :param editable_model: model to be edited
    :param X_test: data for quality evaluaton
    :param y_test: targets for quality evaluaton
    :param X_edit: sequence of data for training model on
    :param y_edit: sequence of targets for training model on
    :param error_function: function that measures quality
    :param progressbar: iterator(range) that may or may not print progress, use progressbar=True for tqdm.tqdm
    :param kwargs: extra parameters for model.edit
    :return: list of results of experiments
    """
    progressbar = tqdm if progressbar is True else progressbar or nop
    results_temporary = []
    with training_mode(editable_model, is_train=False):
        for i in progressbar(range(len(X_edit))):
            edited_model, success, loss, complexity = editable_model.edit(
                X_edit[i:i + 1], y_edit[i:i + 1], detach=True, **kwargs)
            results_temporary.append((error_function(edited_model, X_test, y_test), success, complexity))
    return results_temporary


def evaluate_quality(editable_model: Editable, X_test, y_test, X_edit, y_edit,
                     error_function=classification_error, progressbar=None, **kwargs):
    """
    For each sample in X_edit, y_edit attempts to train model and evaluates trained model quality
    :param editable_model: model to be edited
    :param X_test: data for quality evaluaton
    :param y_test: targets for quality evaluaton
    :param X_edit: sequence of data for training model on
    :param y_edit: sequence of targets for training model on
    :param error_function: function that measures quality
    :param kwargs: extra parameters for model.edit
    :return: dictionary of metrics
    """
    base_error = error_function(editable_model, X_test, y_test)
    results_temporary = calculate_edit_statistics(editable_model, X_test, y_test, X_edit, y_edit,
                                                  progressbar=progressbar, error_function=error_function, **kwargs)
    errors, succeses, complexities = zip(*results_temporary)
    drawdown = np.mean(errors) - base_error
    return dict(base_error=base_error, drawdown=drawdown, success_rate=np.mean(succeses),
                mean_complexity=np.mean(complexities))
