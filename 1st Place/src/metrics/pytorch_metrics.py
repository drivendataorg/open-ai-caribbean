from torch.nn import Module
import torch


def categorical_accuracy(y_pred, y_true):
    max_vals, max_indices = torch.max(y_pred, 1)
    train_acc = (max_indices == y_true).data.to('cpu').numpy().mean()
    return train_acc


class CategoricalAccuracy(Module):

    def __init__(self):
        super(CategoricalAccuracy, self).__init__()
        self.__name__ = 'Accuracy'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        return categorical_accuracy(y_pred, y_true)


def top_k_accuracy(y_pred, y_true, top_k=1):
    dist, ranks = torch.topk(y_pred, top_k)
    y_comp = torch.transpose(y_true.repeat(top_k).view(top_k, -1), 0, 1)
    train_acc = (ranks == y_comp).float().sum(axis=1).mean()
    return train_acc


class TopKAccuracy(Module):

    def __init__(self, top_k=1):
        super(TopKAccuracy, self).__init__()
        self.__name__ = f'Top-{top_k} Accu.'
        self.pattern = '{:.3f}'
        self.top_k = top_k

    def forward(self, y_pred, y_true):
        return top_k_accuracy(y_pred, y_true, self.top_k)


class CrossEntropyLoss(Module):

    def __init__(self, one_hot_encoding=False, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.__name__ = f'CrossEntropyLoss'
        self.pattern = '{:.4f}'
        self.one_hot_encoding = one_hot_encoding
        self.func = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, y_pred, y_true):
        if self.one_hot_encoding:
            _, y_true = y_true.max(dim=-1)
        return self.func(y_pred, y_true)
