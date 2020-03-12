import os
import pydoc
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

tqdm.monitor_interval = 0


class Metrics:
    def __init__(self, functions):
        self.functions = functions
        self.best_score = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class MetricsCollector:
    def __init__(self):
        self.train = []
        self.valid = []

    def update(self, report, is_train):
        if is_train:
            self.train.append(report)
        else:
            self.valid.append(report)


class Runner:
    def __init__(self, factory, callbacks, stages, device):
        self.stages = stages
        self.factory = factory
        self.device = device
        self.model = self.factory.make_model()

        self.model = nn.DataParallel(self.model).to(device)
        self.loss = self.factory.make_loss().to(device)
        self.metrics = Metrics(self.factory.make_metrics())
        self.metrics_collector = MetricsCollector()

        self.current_stage = None
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None

        self.callbacks = callbacks
        self.callbacks.set_trainer(self)
        self.step = 0

        self.accumulation = self.factory.params['accumulation']

    def fit(self, data_factory):
        self.callbacks.on_train_begin()
        for i, stage in enumerate(self.stages):
            print('\n New stage was started')
            self.current_stage = stage

            if 'change_loss' in self.factory.params and self.factory.params['change_loss']:
                self.loss = pydoc.locate(self.current_stage['loss'])(**self.current_stage['loss_params']).to(self.device)

            train_loader = data_factory.make_loader(stage, is_train=True)
            val_loader = data_factory.make_loader(stage, is_train=False)

            if i>0 and self.current_stage['load_best']:
                weights_path = [os.path.join(self.factory.params['save_dir'], w) for w in os.listdir(self.factory.params['save_dir']) if self.factory.params['name_save'] in w]
                if len(weights_path) == 1:
                    weights_path = weights_path[0]
                    print(weights_path)
                    model_name = self.factory.params['model']
                    model = pydoc.locate(model_name)(**self.factory.params['model_params'])
                    model.load_state_dict(torch.load(weights_path)['state_dict'])
                    self.model = nn.DataParallel(model).to(self.device)
                    print('Best checkpoint from previous stage was loaded')

            self.optimizer = self.factory.make_optimizer(self.model, stage)
            if self.current_stage['scheduler'] == 'Cycle_LR':
                self.current_stage['scheduler_params']['optimizer'] = self.optimizer
                self.scheduler = Cycle_LR(**self.current_stage['scheduler_params'])
            else:
                self.scheduler = self.factory.make_scheduler(self.optimizer, stage)

            self.callbacks.on_stage_begin()
            self._run_one_stage(train_loader, val_loader)
            self.callbacks.on_stage_end()

        self.callbacks.on_train_end()

        report = {
            'train_metrics': self.metrics_collector.train,
            'val_metrics': self.metrics_collector.valid
        }

        return report

    def _run_one_stage(self, train_loader, val_loader=None):
        for epoch in range(self.current_stage['epochs']):
            self.callbacks.on_epoch_begin(self.global_epoch)

            self.model.train()
            report = self._run_one_epoch(epoch, train_loader, is_train=True)
            self.metrics.train_metrics = report
            self.metrics_collector.update(report, is_train=True)

            self.model.eval()
            report = self._run_one_epoch(epoch, val_loader, is_train=False)
            self.metrics.val_metrics = report
            self.metrics_collector.update(report, is_train=False)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.val_metrics[str(self.factory.params['metrics'][0])], epoch)
            else:
                self.scheduler.step(epoch)

            self.callbacks.on_epoch_end(self.global_epoch)
            self.global_epoch += 1

    def _run_one_epoch(self, epoch, loader, is_train=True):
        self.step = 0
        epoch_report = defaultdict(float)

        if is_train:
            progress_bar = tqdm(
                enumerate(loader), total=self.factory.params['steps_per_epoch'],
                desc=f"Epoch {epoch} training...", ncols=0)
        else:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch} validating...", ncols=0)

        with torch.set_grad_enabled(is_train):
            len_loader = 0
            for i, data in progress_bar:
                len_loader += 1
                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)
                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)

                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value

                progress_bar.set_postfix(
                    **{k: "{:.5f}".format(v / (i + 1)) for k, v in epoch_report.items()})

                if is_train and i >= self.factory.params['steps_per_epoch']:
                    break
        return {key: value / len_loader for key, value in epoch_report.items()}

    def _make_step(self, data, is_train):
        report = {}
        data = self.batch2device(data)
        images = data['image']
        labels = data['mask']
        
        if is_train and (self.step == 0):
            self.optimizer.zero_grad()

        predictions = self.model(images)
        loss = self.loss(predictions, labels)
        loss = loss / self.accumulation
        report['loss'] = loss.data

        for metric, f in self.metrics.functions.items():
            report[metric] = f(predictions, labels)

        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            report['grad'] = grad_norm
            if self.step % self.accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.step += 1
        return report

    def batch2device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}
