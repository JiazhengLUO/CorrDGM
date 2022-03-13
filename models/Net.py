import torch
import torch.nn as nn
import math
from models.dgcnn import DGCNN
from models.gconv import Siamese_Gconv
from models.affinity_layer import Affinity
import torch.nn.functional as F
from utils.config import cfg


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def make_Adjacency_matrix(x, theta = 1):
    inner = -2 * torch.matmul(x, x.transpose(2, 1).contiguous())
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    A = pairwise_distance/theta  # (batch_size, num_points, k)
    A = torch.exp(A)
    return A


class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x


class norm(nn.Module):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True)
        std = torch.std(x, self.axis,keepdim=True)
        x = (x-mean)/(std+1e-6)
        return x

class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pointfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
        self.gnn_layer = cfg.PGM.GNN_LAYER
        self.add_module('gnn_layer_0', Siamese_Gconv(cfg.PGM.FEATURE_NODE_CHANNEL + cfg.PGM.FEATURE_EDGE_CHANNEL, cfg.PGM.GNN_FEAT))
        self.add_module('affinity_0', Affinity(cfg.PGM.GNN_FEAT))
        self.add_module('InstNorm_layer_0', nn.InstanceNorm2d(1, affine=True))
        self.add_module('affinity_0', Affinity(cfg.PGM.GNN_FEAT))


    # @profile
    def forward(self, P_src, P_tgt ):
        # extract feature
        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()

        theta = 1
        A_src1 = make_Adjacency_matrix(P_src, theta)
        A_tgt1 = make_Adjacency_matrix(P_tgt, theta)

        emb_src, emb_tgt = self.gnn_layer_0([A_src1, emb_src], [A_tgt1, emb_tgt])
        s = self.affinity_0(emb_src, emb_tgt)
        s = self.InstNorm_layer_0(s[:, None, :, :]).squeeze(dim=1)

        log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.PGM.SKADDCR)
        s = torch.exp(log_s)

        return s

