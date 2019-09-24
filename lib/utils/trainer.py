"""
Generalized & extendable trainer class that handles training and evaluation
"""
import os
import time
import glob
from itertools import count, chain
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils import get_latest_file, check_numpy, process_in_chunks, training_mode, \
    infer_model_device, iterate_minibatches, nop_ctx, nop, clear_output
from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy
from tensorboardX import SummaryWriter


class BaseTrainer(nn.Module):
    def __init__(self, model: nn.Module, experiment_name=None, warm_start=False, verbose=False,
                 num_averaged_checkpoints=1, keep_checkpoints=None, **extra_attrs):
        """
        Training helper that trains the model to minimize loss in a supervised mode,
        computes metrics and does a few other tricks if you ask nicely
        :param experiment_name: a path where all logs and checkpoints are saved
        :param warm_start: when set to True, loads last checpoint
        :param verbose: logging verbosity
        :param num_averaged_checkpoints: if > 1, averages this many previous model checkpoints for evaluation
        :param verbose: when set to True, produces logging information
        :param extra_attrs: dict {name: module} to be saved inside trainer via setattr
        """
        super().__init__()
        self.keep_checkpoints = keep_checkpoints or num_averaged_checkpoints
        self.num_averaged_checkpoints = num_averaged_checkpoints
        self.verbose = verbose
        self.total_steps = 0
        self.model = model
        self.best_metrics = {}
        for module_name, module in extra_attrs.items():
            setattr(self, module_name, module)

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(*time.gmtime()[:6])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)

        self.experiment_path = os.path.join('logs/', experiment_name)
        if not warm_start and experiment_name != 'debug':
            assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)
        if warm_start:
            self.load_checkpoint()

    def train_on_batch(self, *args, **kwargs):
        """ Perform a single gradient update and reports metrics and increment self.step """
        raise NotImplementedError()

    def evaluate_metrics(self, *args, **kwargs):
        """ Predicts and evaluates metrics over the entire dataset """
        raise NotImplementedError()

    def predict(self, *inputs, batch_size=1024, is_train=False, device=None, memory_efficient=False, **kwargs):
        """
        Compute model predictions over a (large) number of samples
        :param inputs: one or several input arrays, each of shape [batch_size, *whatever]
        :param batch_size: predicts for this many samples over one call to the model
        :param is_train: if True, runs model in training mode (e.g. with dropout)
        :param device: moves all inputs to that device, defaults to infer_model_device
        :param memory_efficient: if True, data is transferred to device one batch at a time
            otherwise (default), transfers all data on device in advance
        :param kwargs: key-value arguments passed to every model call
        :return:
        """
        inputs = tuple(map(torch.as_tensor, inputs))
        device = device or infer_model_device(self.model)

        if memory_efficient:
            def predict_on_batch(*batch):
                batch = (tensor.to(device=device) for tensor in batch)
                return self.model(*batch, **kwargs).cpu()
        else:
            inputs = (tensor.to(device=device) for tensor in inputs)
            predict_on_batch = self.model

        with training_mode(self.model, is_train=is_train), torch.no_grad():
            predictions = process_in_chunks(predict_on_batch, *inputs, batch_size=batch_size)
            predictions = check_numpy(predictions)
        return predictions

    def record(self, *, prefix='', **metrics):
        """
        Computes and saves metrics into tensorboard
        :param prefix: common prefix for tensorboard
        :param metrics: key-value parameters forwarded into every metric
        :return: metrics (same as input)
        """
        if not (prefix == '' or prefix.endswith('/')):
            warn("It is recommended that prefix ends with slash(/) for readability")

        for key, value in metrics.items():
            assert np.shape(value) == (), "metric {} must be scalar, but got {}".format(key, np.shape(value))
            self.writer.add_scalar(prefix + str(key), value, self.total_steps)
        return metrics

    def save_checkpoint(self, tag=None, path=None, mkdir=True, clear_old=False, number_ckpts_to_keep=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = "temp_{}".format(self.total_steps)
        if path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.total_steps),
            ('best_metrics', self.best_metrics),
        ]), path)
        if self.verbose:
            print("Saved " + path)
        if clear_old:
            self.remove_old_temp_checkpoints(number_ckpts_to_keep)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        elif tag is not None and path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)
        self.opt.load_state_dict(checkpoint['opt'])
        self.total_steps = int(checkpoint['step'])
        self.best_metrics = checkpoint['best_metrics']

        if self.verbose:
            print('Loaded ' + path)
        return self

    @contextmanager
    def using_checkpoint(self, **kwargs):
        """
        Backups current checkpoint, loads new one in context, restores current checkpoint upon exiting context
        :param kwargs: loads checkpoint with these params (e.g. tag or path)
        """
        current_checkpoint_tag = 'current'
        while True:
            current_checkpoint_tag += '_backup'
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(current_checkpoint_tag))
            if not os.path.exists(path):
                break

        self.save_checkpoint(current_checkpoint_tag)
        self.load_checkpoint(**kwargs)
        yield
        self.load_checkpoint(current_checkpoint_tag)
        os.remove(path)

    def average_checkpoints(self, tags=None, paths=None, out_tag='avg', out_path=None):
        assert tags is None or paths is None, "please provide either tags or paths or nothing, not both"
        assert out_tag is not None or out_path is not None, "please provide either out_tag or out_path or both"
        if tags is None and paths is None:
            paths = self.get_latest_checkpoints(
                os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'), self.num_averaged_checkpoints)
        elif tags is not None and paths is None:
            paths = [os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(tag)) for tag in tags]

        checkpoints = [torch.load(path) for path in paths]
        averaged_ckpt = deepcopy(checkpoints[0])
        for key in averaged_ckpt['model']:
            values = [ckpt['model'][key] for ckpt in checkpoints]
            averaged_ckpt['model'][key] = sum(values) / len(values)

        if out_path is None:
            out_path = os.path.join(self.experiment_path, 'checkpoint_{}.pth'.format(out_tag))
        torch.save(averaged_ckpt, out_path)

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        assert len(list_of_files) > 0, "No files found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.keep_checkpoints
        paths = self.get_latest_checkpoints(os.path.join(self.experiment_path, 'checkpoint_temp_[0-9]*.pth'))
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            if self.verbose:
                print("Removing", ckpt)
            os.remove(ckpt)

    def step(self, *args, **kwargs):
        """ Trains on batch and updates the counter of total_steps """
        was_steps = self.total_steps
        metrics = self.train_on_batch(*args, **kwargs)
        assert self.total_steps == was_steps, "total_steps changed within train_on_batch"
        self.total_steps += 1
        return metrics

    def forward(self, *inputs, **kwargs):
        """ see train_on_batch """
        return self.step(*inputs, **kwargs)

    def fit(self, training_data, batch_size=None, shuffle=True, epochs=1, start_epoch=1, batches_per_epoch=None,
            batcher_kwargs=None, progressbar=None, clear_outputs=False, device='auto', val_data=None, eval_kwargs=None,
            early_stopping_minimize=(), early_stopping_maximize=(), early_stopping_epochs=None, **kwargs):
        """
        Trains for one or several epochs on minibatches of data, optionally evaluates dev metrics after each epoch
        :param training_data: training data source, must be either of
            * torch DataLoader or Dataset
            * a list or tuple of tensors
            * iterator of minibatches
        :param batch_size: splits tensors into chunks of this size over 0-th divension
        :param shuffle: (default) shuffles tensors symmetrically over 0-th dimension
        :param batcher_kwargs: keyword parameters to be fed into data iterator
        :param progressbar: if True or callback (e.g. tqdm), prints progress of each training epoch
        :param epochs: performs this many passes over training data, float('inf') for inifinite loop
        :param device: puts minibatches on this device. None to keep original device, 'auto' to try infer model device,
        :param val_data: if not None, calls self.evaluate_metrics on this data after each epoch
        :param eval_kwargs: additional kwargs for self.evaluate_metrics, only used if eval_data is not None
        :param early_stopping_maximize: keeps checkpoints with highest values of these metrics
        :param early_stopping_minimize: keeps checkpoints with lowest values of these metrics
        :param early_stopping_epochs: stops training if there were no updates on early_stopping_maximize/minimize
            for at least this many epochs
        :param start_epoch: initial epoch index, only used for printing (epoch ##)
        :param kwargs: additional kwargs for self.step (train_on_batch)
        :return: self
        """
        device = getattr(self, 'device', infer_model_device(self)) if device == 'auto' else device
        progressbar = tqdm if progressbar is True else progressbar or nop
        epochs, early_stopping_epochs = epochs or float('inf'), early_stopping_epochs or float('inf')
        eval_kwargs, batcher_kwargs = eval_kwargs or dict(), batcher_kwargs or dict()
        if isinstance(early_stopping_minimize, str): early_stopping_minimize = [early_stopping_minimize]
        if isinstance(early_stopping_maximize, str): early_stopping_maximize = [early_stopping_maximize]
        number_of_epochs_without_improvement = 0

        # prepare training data one way or another
        if isinstance(training_data, DataLoader):
            make_training_epoch = lambda: iter(progressbar(training_data))
        elif isinstance(training_data, Dataset):
            make_training_epoch = torch.utils.data.DataLoader(
                training_data, batch_size=batch_size, shuffle=shuffle, **batcher_kwargs)
        elif isinstance(training_data, (list, tuple)):
            make_training_epoch = lambda: iterate_minibatches(
                *training_data, batch_size=batch_size, epochs=1, shuffle=shuffle,
                callback=progressbar, **batcher_kwargs)
        else:
            training_data = iter(training_data)
            assert batches_per_epoch is not None or epochs == 1, "if data is an iterator, please provide " \
                                                                 "batches_per_epoch or use a single epoch"
            def make_training_epoch():
                for _ in progressbar(range(batches_per_epoch) if batches_per_epoch else count()):
                    yield next(training_data)

        # iterate training epochs
        for epoch_i in count(start=start_epoch):
            if epoch_i >= epochs + start_epoch:
                if self.verbose:
                    print("Stopping because of reaching target number of epochs")
                break
            if self.verbose:
                print("Epoch #{}/{}".format(epoch_i, epochs))

            for batch in make_training_epoch():
                if device is not None:
                    batch = tuple(torch.as_tensor(tensor, device=device) for tensor in batch)
                self.step(*batch, **kwargs)

            if clear_outputs:
                clear_output()

            self.save_checkpoint(clear_old=True)
            if self.num_averaged_checkpoints > 1:
                self.average_checkpoints(out_tag='avg')

            if val_data is not None:
                if self.verbose:
                    print("Evaluating...")

                with self.using_checkpoint(tag='avg') if self.num_averaged_checkpoints > 1 else nop_ctx():
                    val_metrics = self.evaluate_metrics(*val_data, **eval_kwargs)

                if self.verbose:
                    for key, value in val_metrics.items():
                        print(key, value)
                    print()

                # handle best metrics and early stopping
                number_of_epochs_without_improvement += 1

                for key, value in val_metrics.items():
                    found_new_best = False
                    if key in early_stopping_maximize:
                        if value > self.best_metrics.get(key, -float('inf')):
                            found_new_best = True
                    if key in early_stopping_minimize:
                        if value < self.best_metrics.get(key, float('inf')):
                            found_new_best = True
                    if found_new_best:
                        self.best_metrics[key] = value
                        number_of_epochs_without_improvement = 0
                        self.save_checkpoint(tag='best_' + key)

                for key in chain(early_stopping_maximize, early_stopping_minimize):
                    if key not in val_metrics:
                        warn("Metric name {} not found but requested for maximizing/minimizing")

                if number_of_epochs_without_improvement >= early_stopping_epochs:
                    if self.verbose:
                        print("Early stopping because of no improvement in "
                              "{} epochs".format(number_of_epochs_without_improvement))
                    break

            else:
                assert eval_kwargs is None, "Eval kwargs is unused if val_data is None"
                assert early_stopping_epochs == float('inf'), "Early stopping requires val_data"
                assert len(early_stopping_minimize) == len(early_stopping_maximize) == 0, "Please provide val_data"

        return self


class SupervisedTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, loss_function, opt=None, **kwargs):
        """ A simple optimizer that trains to minimize classification or regression loss """
        opt = opt if opt is not None else torch.optim.Adam(model.parameters())
        super().__init__(model, loss_function=loss_function, opt=opt, **kwargs)

    def train_on_batch(self, x_batch, y_batch, prefix='train/', is_train=True):
        """ Performs a single gradient update and reports metrics """
        x_batch, y_batch = map(torch.as_tensor, (x_batch, y_batch))
        self.opt.zero_grad()

        with training_mode(self.model, is_train=is_train):
            prediction = self.model(x_batch)

        loss = self.loss_function(prediction, y_batch).mean()
        loss.backward()
        self.opt.step()

        return self.record(loss=loss.item(), prefix=prefix)

    def evaluate_metrics(self, X, y, prefix='val/', **kwargs):
        """ Predicts and evaluates metrics over the entire dataset """
        prediction = self.predict(X, **kwargs)
        with torch.no_grad():
            loss = self.loss_function(torch.as_tensor(prediction), torch.as_tensor(y)).mean()

        return self.record(loss=loss.item(), prefix=prefix)
