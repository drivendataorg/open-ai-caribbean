from torch.optim.lr_scheduler import StepLR as pt_StepLR


class StepLR(pt_StepLR):

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, min_lr=0):

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        super(StepLR, self).__init__(optimizer, step_size, gamma=0.1, last_epoch=-1)

    def get_lr(self):

        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]

    def step(self, epoch=None, **kwargs):

        super(StepLR, self).step(epoch)
