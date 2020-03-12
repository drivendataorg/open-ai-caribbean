import logging
import os
from copy import deepcopy

import torch
from tensorboardX import SummaryWriter
from youtrain.utils import get_last_save


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics = None

    def set_trainer(self, runner):
        self.runner = runner
        self.metrics = runner.metrics

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_stage_begin(self):
        pass

    def on_stage_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def set_trainer(self, runner):
        for callback in self.callbacks:
            callback.set_trainer(runner)

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_stage_begin(self):
        for callback in self.callbacks:
            callback.on_stage_begin()

    def on_stage_end(self):
        for callback in self.callbacks:
            callback.on_stage_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class ModelSaver(Callback):
    def __init__(
            self, save_dir, save_every, save_name,
            best_only=True, metric_name='loss', checkpoint=True, threshold=0.3):
        super().__init__()
        self.checkpoint = checkpoint
        self.metric_name = metric_name
        self.save_dir = save_dir
        self.save_every = save_every
        self.threshold = threshold

        self.save_name = save_name
        self.best_only = best_only
        self.current_path = None

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.current_path = str(self.save_dir / self.save_name) + '_0.0'

    def save_checkpoint(self, epoch, path, score=None):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.runner.model.module.state_dict(),
            'optimizer': self.runner.optimizer.state_dict(),
            'best_score': self.metrics.best_score}, path)
        print(f'Model was saved at {self.save_name} with score {score}')

    def on_epoch_end(self, epoch):
        score = float(self.metrics.val_metrics[self.metric_name])
        need_save = not self.best_only
        if epoch % self.save_every == 0:
            if score < self.metrics.best_score:
                self.metrics.best_score = score
                self.metrics.best_epoch = epoch
                need_save = True

            if need_save:
                if os.path.exists(self.current_path + '.pt'):
                    current_score = float(self.current_path.split('_')[-1])
                    if current_score < self.threshold:
                        os.remove(self.current_path + '.pt')
                self.current_path = '_'.join(self.current_path.split('_')[:-1]) + '_{:.5f}'.format(abs(self.metrics.best_score))
                if self.checkpoint:
                    self.save_checkpoint(epoch=epoch, path=self.current_path + '.pt', score=score)
                else:
                    torch.save(obj=deepcopy(self.runner.model.module), f=self.current_path + '.pt')
                    print(f'Model was saved at {self.save_name} with score {score}')


class TensorBoard(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch):
        for k, v in self.metrics.train_metrics.items():
            self.writer.add_scalar(f'train/{k}', float(v), global_step=epoch)

        for k, v in self.metrics.val_metrics.items():
            self.writer.add_scalar(f'val/{k}', float(v), global_step=epoch)

        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'group{idx}/lr', float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer.close()


class Logger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.logger = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._get_logger(str(self.log_dir / 'logs.txt'))
        self.logger.info(f'Starting training with params:\n{self.runner.factory.params}\n\n')

    def on_epoch_begin(self, epoch):
        self.logger.info(
            f'Epoch {epoch} | '
            f'optimizer "{self.runner.optimizer.__class__.__name__}" | '
            f'lr {self.current_lr}')

    def on_epoch_end(self, epoch):
        self.logger.info(
            "Train metrics: " + self._get_metrics_string(self.metrics.train_metrics))
        self.logger.info(
            "Valid metrics: " + self._get_metrics_string(self.metrics.val_metrics) + "\n")

    def on_stage_begin(self):
        self.logger.info(f'Starting stage:\n{self.runner.current_stage}\n')

    @staticmethod
    def _get_logger(log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        res = []
        for param_group in self.runner.optimizer.param_groups:
            res.append(param_group['lr'])
        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _get_metrics_string(metrics):
        return " | ".join("{}: {:.5f}".format(k, v) for k, v in metrics.items())


class OneCycleLR(Callback):
    """
    An learning rate updater
        that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.

    https://github.com/Scitator/pytorch-common/blob/master/train/callbacks.py
    """

    def __init__(self, init_lr, cycle_len, div, cut_div, momentum_range, len_loader):
        """
        :param init_lr: init learning rate for torch optimizer
        :param cycle_len: (int) num epochs to apply one cycle policy
        :param div: (int) ratio between initial lr and maximum lr
        :param cut_div: (int) which part of cycle lr will grow
            (Ex: cut_div=4 -> 1/4 lr grow, 3/4 lr decrease
        :param momentum_range: (tuple(int, int)) max and min momentum values
        """
        super().__init__()
        self.init_lr = init_lr
        self.len_loader = len_loader
        self.total_iter = None
        self.div = div
        self.cut_div = cut_div
        self.cycle_iter = 0
        self.cycle_count = 0
        self.cycle_len = cycle_len
        # point in iterations for starting lr decreasing
        self.cut_point = None
        self.momentum_range = momentum_range

    def calc_lr(self):
        # calculate percent for learning rate change
        if self.cycle_iter > self.cut_point:
            percent = 1 - (self.cycle_iter - self.cut_point) / (
                    self.total_iter - self.cut_point)
        else:
            percent = self.cycle_iter / self.cut_point
        res = self.init_lr * (1 + percent * (self.div - 1)) / self.div

        self.cycle_iter += 1
        if self.cycle_iter == self.total_iter:
            self.cycle_iter = 0
            self.cycle_count += 1
        return res

    def calc_momentum(self):
        if self.cycle_iter > self.cut_point:
            percent = (self.cycle_iter - self.cut_point) / (self.total_iter - self.cut_point)
        else:
            percent = 1 - self.cycle_iter / self.cut_point
        res = self.momentum_range[1] + percent * (
                self.momentum_range[0] - self.momentum_range[1])
        return res

    def update_lr(self, optimizer):
        new_lr = self.calc_lr()
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def update_momentum(self, optimizer):
        new_momentum = self.calc_momentum()
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum
        return new_momentum

    def on_batch_end(self, i, **kwargs):
        if kwargs['is_train']:
            self.update_lr(self.runner.optimizer)
            self.update_momentum(self.runner.optimizer)

    def on_train_begin(self):
        self.total_iter = self.len_loader * self.cycle_len
        self.cut_point = self.total_iter // self.cut_div

        self.update_lr(self.runner.optimizer)
        self.update_momentum(self.runner.optimizer)


class LRFinder(Callback):
    """
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, len_loader, init_lr, final_lr, beta, save_name):
        super().__init__()
        self.save_name = save_name
        self.beta = beta
        self.final_lr = final_lr
        self.init_lr = init_lr
        self.len_loader = len_loader
        self.multiplier = (self.final_lr / self.init_lr) ** (1 / self.len_loader)
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.find_iter = 0
        self.losses = []
        self.log_lrs = []
        self.is_find = False

    def calc_lr(self):
        res = self.init_lr * self.multiplier ** self.find_iter
        self.find_iter += 1
        return res

    def update_lr(self, optimizer):
        new_lr = self.calc_lr()
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        return new_lr

    def on_batch_end(self, i, **kwargs):
        loss = kwargs['step_report']['loss'].item()
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** self.find_iter)

        if smoothed_loss < self.best_loss or self.find_iter == 1:
            self.best_loss = smoothed_loss

        if not self.is_find:
            self.losses.append(smoothed_loss)
            self.log_lrs.append(self.update_lr(self.runner.optimizer))

        if self.find_iter > 1 and smoothed_loss > 4 * self.best_loss:
            self.is_find = True

    def on_train_begin(self):
        self.update_lr(self.runner.optimizer)

    def on_train_end(self):
        torch.save({
            'best_loss': self.best_loss,
            'log_lrs': self.log_lrs,
            'losses': self.losses,
        }, os.path.join(self.runner.model_dir, self.save_name))
