import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class Fbeta_score(nn.Module):
    def __init__(self, beta=2, threshold=0.3):
        super(Fbeta_score, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def fbeta_score(self, y_true, y_pred, beta, threshold, eps=1e-9):
        beta2 = beta ** 2

        y_pred = torch.ge(y_pred.float(), threshold).float()
        y_true = y_true.float()

        true_positive = (y_pred * y_true).sum(dim=1)
        precision = true_positive.div(y_pred.sum(dim=1).add(eps))
        recall = true_positive.div(y_true.sum(dim=1).add(eps))

        return torch.mean(
            (precision * recall).
                div(precision.mul(beta2) + recall + eps).
                mul(1 + beta2))

    def forward(self, preds, targs):
        return -self.fbeta_score(targs, preds.sigmoid(), self.beta, self.threshold)


class Dice(nn.Module):
    def __init__(self,):
        super(Dice, self).__init__()

    def forward(self, outputs, targets):
        smooth = 1e-15
        target = (targets > 0.0).float()
        prediction = F.sigmoid(outputs)
        dice_part = (2*torch.sum(prediction * target) + smooth) / \
                            (torch.sum(prediction) + torch.sum(target) + smooth)
        return -dice_part


class MultiAccuracy(nn.Module):
    def __init__(self):
        super(MultiAccuracy, self).__init__()

    def forward(self, outputs, targets):
        prediction = outputs.argmax(dim=1)
        prediction = prediction.cpu().detach().numpy().flatten()
        target = targets.cpu().detach().numpy().flatten()
        return -np.mean(prediction == target)

