import time
import copy
import sys

import numpy as np

import torch
from torch import nn

from src.utils.misc import RunningAverage, htime


class PyTorchNN_vA(object):

    def __init__(self, model, optimizer=None, criterion=None, metrics=None, scheduler=None, early_stopper=None,
                 device='auto', virtual_batch_size=None, mixup=False, mixup_alpha=1, mixup_method='default'):

        # Model
        self.model = model

        # Optimizer
        self.optimizer = optimizer

        # Criterion
        self.criterion = criterion

        # Metrics
        self.metrics_names = [criterion.__class__.__name__, ]
        self.metrics = metrics
        self.metrics_names = self.metrics_names + [s.__class__.__name__ for s in self.metrics] \
            if self.metrics is not None else self.metrics_names

        # Scheduler
        if scheduler is not None:
            self.scheduler = scheduler if isinstance(scheduler, list) else [scheduler, ]
        else:
            self.scheduler = scheduler

        # Early stopper
        self.early_stopper = early_stopper

        # Device. If 'auto', use cuda when available
        if device == 'auto':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Batch size
        self.virtual_batch_size = virtual_batch_size
        
        # Mixup
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        assert all([s in ['default', 'losses', 'inbatch'] for s in mixup_method.split('|')])
        self.mixup_method= mixup_method
         

    def clean_memory(self):
        self.model.cpu()
        self.model = None

    def _iterate_loader(self, data_loader, phase='train'):

        assert phase in ['train', 'valid', 'predict']
        if phase == 'train':
            init_txt = "Training | "
            training = True
            evaluate = True
            return_p = False
            self.model.train()
        elif phase == 'valid':
            init_txt = "Validation | "
            training = False
            evaluate = True
            return_p = False
            self.model.eval()
        elif phase == 'predict':
            init_txt = "Predicting | "
            training = False
            evaluate = False
            return_p = True
            self.model.eval()
        else:
            raise ValueError(f"Not recognised phase: {phase}")

        # Initiate variables
        loss_sofar = RunningAverage()
        if self.metrics is not None:
            metric_loss_so_far = [RunningAverage() for _ in self.metrics]
        total_batches = len(data_loader)
        count_batches = 0
        accum_batch_size = 0
        accum_loss = 0
        since = time.time()
        batch_time = time.time()
        first_print = True
        preds = []

        for batch_nb, data in enumerate(data_loader):

            # Get inputs
            inputs, labels = data
            
            add_losses = False
            if self.mixup and training:
                # TODO: Make it compatible with multi-input
                bs = inputs.shape[0]
                
                if 'inbatch' in self.mixup_method:
                    lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, size=bs)).float()
                    new_bs = np.arange(bs)
                    np.random.shuffle(new_bs)
                    inputs = lam.view((bs, 1, 1, 1)) * inputs + (1 - lam.view((bs, 1, 1, 1))) * inputs[new_bs]
                else:
                    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    inputs = (lam*inputs[0:bs//2] + (1-lam)*inputs[bs//2:(bs//2)*2])
                    
                if len(labels.size()) > 1: #Assume BCE
                    if 'inbatch' in self.mixup_method:
                        raise NotImplementedError
                    else:
                        labels = (lam*labels[0:bs//2] + (1-lam)*labels[bs//2:(bs//2)*2])
                        
                else:  # Assume CategoricalCrossEntropy
                    if 'default' in self.mixup_method:
                        if 'inbatch' in self.mixup_method:
                            labels[lam < 0.5] = labels[new_bs][lam < 0.5]
                        else:
                            if lam < 0.5:
                                labels = labels[bs//2:(bs//2)*2]
                            else:
                                labels = labels[0:bs//2]
                                
                    elif 'losses' in self.mixup_method:
                        add_losses = True
                        labels = labels

            # Multi imputs
            if isinstance(inputs, list):
                this_batch_size = inputs[0].size(0)
                # Move to device
                inputs = [s.to(self.device) for s in inputs]
            else:
                this_batch_size = inputs.size(0)
                # Move to device
                inputs = inputs.to(self.device)

            # Move to device
            if evaluate:
                labels = labels.to(self.device)

            # zero the parameter gradients
            if training and batch_nb == 0:
                self.optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(training):

                # Get outputs
                outputs = self.model(inputs)

                if return_p:
                    preds.append(outputs.detach().cpu().numpy())

                # Get loss
                if evaluate:
                    if add_losses:
                        if 'inbatch' in self.mixup_method:
                            raise NotImplementedError
                        else:
                            loss = lam * self.criterion(outputs, labels[0:bs//2]) + (1-lam) * self.criterion(outputs, labels[bs//2:(bs//2)*2])
                    else:
                        loss = self.criterion(outputs, labels)
                    loss_sofar.update(this_batch_size, loss.item())

                # backward + optimize only if in training phase
                if training:
                    # Virtual batch
                    accum_batch_size += this_batch_size

                    if self.virtual_batch_size is None:
                        # Normal execution
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accum_batch_size = 0
                    elif batch_nb >= len(data_loader) - 1:
                        # Last batch
                        loss = loss * (this_batch_size/self.virtual_batch_size)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accum_batch_size = 0
                    elif accum_batch_size >= self.virtual_batch_size:
                        # Propagate gradients
                        loss = loss * (this_batch_size / self.virtual_batch_size)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accum_batch_size = 0
                    else:
                        loss = loss * (this_batch_size / self.virtual_batch_size)
                        loss.backward()

            # Calculate metrics
            metrics_txt = ''
            if self.metrics is not None and evaluate:
                metrics_txt += 'Metrics: ['
                for im, metric in enumerate(self.metrics):
                    if add_losses:
                        metric_loss = lam * metric(outputs, labels[0:bs//2]) + (1-lam) * metric(outputs, labels[bs//2:(bs//2)*2])
                    else:
                        metric_loss = metric(outputs, labels)
                    metric_loss_so_far[im].update(this_batch_size, metric_loss.item())
                    # metrics_txt += f'{metric_loss_so_far[im].average():.4f}, '
                    if im > 0:
                        metrics_txt += ', '
                    metrics_txt += f'{metric.__name__}: ' + metric.pattern.format(metric_loss_so_far[im].average())
                metrics_txt += '] '

            # Update loss/metrics
            count_batches += 1

            # Print progress
            if (time.time() - batch_time) > 2 or count_batches >= total_batches:  # Print every 2 second or more
                time_elapsed = time.time() - since
                time_remain = time_elapsed / float(count_batches) * (total_batches - count_batches)
                if not first_print:
                    sys.stdout.write('\r')
                    # sys.stdout.flush()
                mtxt = init_txt + f'Batches:{batch_nb+1}/{total_batches} Samples:{loss_sofar.samples} '
                if evaluate:
                    mtxt += f'Loss: {loss_sofar.average():.5f} '
                    mtxt += metrics_txt
                mtxt += f'(Time: {htime(time_elapsed)} | ETA: {htime(time_remain)}) '
                sys.stdout.write(mtxt)
                sys.stdout.flush()
                batch_time = time.time()
                first_print = False

            # Release data from gpu
            if isinstance(inputs, list):
                inputs = [s.cpu() for s in inputs]
            else:
                inputs = inputs.cpu()
            labels = labels.cpu()
            if evaluate:
                loss = loss.cpu()
                del loss
            del(inputs, labels)
            torch.cuda.empty_cache()

        if return_p:
            return np.vstack(preds)

        return loss_sofar.average()

    def train_loader(self, data_loaders, epochs, initial_epoch=0):

        # init
        since = time.time()
        self.model = self.model.to(self.device)
        num_epochs = epochs - initial_epoch
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 100

        # Train epochs
        for epoch in range(initial_epoch, epochs):
            since_epoch = time.time()
            epoch_txt = 'Epoch {}/{}'.format(epoch + 1, epochs)
            if self.scheduler is not None:
                epoch_txt += f" LR: {[s1['lr'] for s1 in self.optimizer.param_groups]}"
            print(epoch_txt)
            print('-' * len(epoch_txt))

            # https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()

            # Train
            train_loss = None
            if 'train' in data_loaders:
                train_loss = self._iterate_loader(data_loaders['train'], phase='train')

            print('')

            # Validation
            valid_loss = None
            if 'valid' in data_loaders:
                valid_loss = self._iterate_loader(data_loaders['valid'], phase='valid')
                print('')

                # Save best model
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # Update early_stopper
                if self.early_stopper is not None:
                    self.early_stopper(valid_loss, self.model)

                    if self.early_stopper.early_stop:
                        print("Early stopping")
                        break

            # Upadte shedulers
            if self.scheduler is not None:
                for sch in self.scheduler:
                    sch.step(metrics=valid_loss) if valid_loss is not None else sch.step(metrics=train_loss)

            time_elapsed = time.time() - since_epoch
            print(f'Epoch complete in {htime(time_elapsed)}')
            #print(GPUtil.showUtilization())
            print('')

        time_elapsed = time.time() - since
        print(f'Training complete in {htime(time_elapsed)}')

        # Load state of best model
        if 'valid' in data_loaders:
            print('Best val loss: {:4f}'.format(best_loss))
            self.model.load_state_dict(best_model_wts)

    def evaluate_loader(self, data_loader):

        # init
        since = time.time()
        self.model = self.model.to(self.device)

        # https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        result = self._iterate_loader(data_loader, phase='valid')

        time_elapsed = time.time() - since
        print(f'Evaluation complete in {htime(time_elapsed)}')

        return result

    def predict_loader(self, data_loader):

        # init
        since = time.time()
        self.model = self.model.to(self.device)

        # https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        # Predict
        if isinstance(self.model, nn.DataParallel):
            self.model.module.output_predictions = True
        else:
            self.model.output_predictions = True

        result = self._iterate_loader(data_loader, phase='predict')

        if isinstance(self.model, nn.DataParallel):
            self.model.module.output_predictions = False
        else:
            self.model.output_predictions = False

        time_elapsed = time.time() - since
        print(f'Prediction complete in {htime(time_elapsed)}')

        return result
