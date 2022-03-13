import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=0.5, min_clamp: Optional[float] = 1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.min_clamp = min_clamp

    def forward(self, pred, target):
        # pred B,N,N target B,N,N
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        pred = torch.cat((1 - pred, pred), dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        class_mask.scatter_(1, target.long(), 1.)
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min = self.min_clamp, max=1.0 - self.min_clamp)
        log_p = probs.log()
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        loss = batch_loss.mean()
        return loss


class Euclidian_loss(nn.Module):
    def __init__(self):
        super(Euclidian_loss, self).__init__()

    def forward(self, per, label):
        distance = (per - label) ** 2
        dis = torch.mean(distance)

        return dis



if __name__ == '__main__':
    d1 = torch.tensor([[1., 2.],
                       [2., 3.],
                       [3., 4.]], requires_grad=True)
    d2 = torch.tensor([[-1., -2.],
                       [-2., -3.],
                       [-3., -4.]], requires_grad=True)
    # mask = torch.tensor([[1., 1.],
    #                      [1., 1.],
    #                      [0., 0.]])

    rl = FocalLoss()
    loss = rl(d1.cuda(), d2.cuda())
    loss.backward()
    print(d1.grad)
    print(d2.grad)
