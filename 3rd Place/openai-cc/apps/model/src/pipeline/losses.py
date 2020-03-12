import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, logits, targets):
        if isinstance(logits, tuple):
            logits = logits[0]
        targets = targets.squeeze(dim=1).type(torch.cuda.LongTensor)
        return self.loss(logits, targets)


class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 0
        for i in range(2):
            probs_flat = probs[:, i].contiguous().view(-1)
            targets_flat = (targets==i+1).float().contiguous().view(-1)
            loss += self.bce_loss(probs_flat, targets_flat)
        return loss


class FbetaLoss(nn.Module):
    def __init__(self, beta=2):
        super(FbetaLoss, self).__init__()
        self.small_value = 1e-6
        self.beta = beta

    def forward(self, logits, labels):
        beta = self.beta
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossBinaryDice, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = 1e-15
            target = (targets > 0.0).float()
            prediction = F.sigmoid(outputs)
            dice_part = 1 - (2*torch.sum(prediction * target) + smooth) / \
                            (torch.sum(prediction) + torch.sum(target) + smooth)
            loss += self.dice_weight * dice_part

        return loss
