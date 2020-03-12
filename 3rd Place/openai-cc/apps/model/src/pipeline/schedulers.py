from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


from torch.optim import Optimizer


class OneCycleLR:
    """ Sets the learing rate of each parameter group by the one cycle learning rate policy
    proposed in https://arxiv.org/pdf/1708.07120.pdf.
    It is recommended that you set the max_lr to be the learning rate that achieves
    the lowest loss in the learning rate range test, and set min_lr to be 1/10 th of max_lr.
    So, the learning rate changes like min_lr -> max_lr -> min_lr -> final_lr,
    where final_lr = min_lr * reduce_factor.
    Note: Currently only supports one parameter group.
    Args:
        optimizer:             (Optimizer) against which we apply this scheduler
        num_steps:             (int) of total number of steps/iterations
        lr_range:              (tuple) of min and max values of learning rate
        momentum_range:        (tuple) of min and max values of momentum
        annihilation_frac:     (float), fracion of steps to annihilate the learning rate
        reduce_factor:         (float), denotes the factor by which we annihilate the learning rate at the end
        last_step:             (int), denotes the last step. Set to -1 to start training from the beginning
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(0.1, 1.))
        >>> for epoch in range(epochs):
        >>>     for step in train_dataloader:
        >>>         train(...)
        >>>         scheduler.step()
    Useful resources:
        https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6
        https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0
    """

    def __init__(self,
                 optimizer: Optimizer,
                 num_steps: int,
                 lr_range: tuple = (0.1, 1.),
                 momentum_range: tuple = (0.85, 0.95),
                 annihilation_frac: float = 0.1,
                 reduce_factor: float = 0.01,
                 last_step: int = -1):
        # Sanity check
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

        self.num_steps = num_steps

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        self.num_cycle_steps = int(num_steps * (1. - annihilation_frac))  # Total number of steps in the cycle
        self.final_lr = self.min_lr * reduce_factor

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    def step(self):
        """Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            # Scale up phase
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            # Annihilation phase: only change lr
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            # Exceeded given num_steps: do nothing
            return

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum



class Cycle_LR(_LRScheduler):
    def __init__(self, optimizer, lr_factor, cycle_len, cycle_factor=2, gamma=None, jump=None, \
                 last_epoch=-1, surface=None, momentum_range=None):
        self.lr_factor = lr_factor
        self.cycle_len = cycle_len
        self.cycle_factor = cycle_factor
        self.gamma = gamma
        self.jump = jump
        self.last_epoch = last_epoch
        self.dec = True
        self.stage_epoch_count = 0
        self.current_cycle = cycle_len
        self.momentum_range = momentum_range
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.surface = surface
        if not surface:
            self.surface = self.surface_f

        for group in self.optimizer.param_groups:
            group['lr'] *= self.lr_factor

        if self.momentum_range:
            for group in self.optimizer.param_groups:
                group['momentum'] = momentum_range[1]

    @staticmethod
    def surface_f(x):
        return(np.tanh(- 2 *( 2 * x -0.8)) +1) / 2

    def switch(self):
        if self.jump:
            if self.dec:
                self.current_cycle = self.jump
                self.cycle_len *= self.cycle_factor
                if self.gamma:
                    self.lr_factor *= self.gamma
            else:
                self.current_cycle = self.cycle_len
            self.dec = not self.dec

    def get_lr(self):
        if self.stage_epoch_count > self.current_cycle:
            self.switch()
            self.stage_epoch_count = 0

        percent = self.stage_epoch_count / self.current_cycle
        if self.dec:
            factor = 1 + self.surface(percent) * (self.lr_factor - 1)
        else:
            factor = 1 + percent * (self.lr_factor - 1)

        self.stage_epoch_count += 1
        return [factor] * len(self.optimizer.param_groups)

    def get_momentum(self):
        percent = self.stage_epoch_count / self.current_cycle
        if self.dec:
            momentum = self.momentum_range[1] + percent * (
                    self.momentum_range[0] - self.momentum_range[1])
        else:
            momentum = self.momentum_range[0] + percent * (
                    self.momentum_range[1] - self.momentum_range[0])
        return [momentum] * len(self.optimizer.param_groups)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = param_group['initial_lr'] * lr

        if self.momentum_range:
            for param_group, m in zip(self.optimizer.param_groups, self.get_momentum()):
                param_group['momentum'] = m


