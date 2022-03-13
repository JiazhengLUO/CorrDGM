import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import SHRECTest, SHRECTest_witout
from models import Net
from tensorboardX import SummaryWriter
from datetime import datetime
from utils.config import cfg
from pathlib import Path
from utils.model_sl import load_model, save_model
import os



def square_distance(src, dst):
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    return dist


def make_soft_label(label_origin, xyz2, ratio=0.5):
    if ratio == 0.0:
        return label_origin
    else:
        label = label_origin.clone()

        dist = square_distance(xyz2, xyz2)

        max_square_radius = torch.max(dist)

        radius = ratio * torch.sqrt(max_square_radius)

        for row in range(label.shape[0]):

            idx = torch.nonzero(label[row])

            dist_row = dist[idx.squeeze()]
            add_idx = (dist_row <= radius ** 2).nonzero().squeeze()

            if add_idx.ndimension() == 0:
                add_idx = [add_idx]

            for i in add_idx:
                label[row][i] = 1.0

        soft_label = label

    return soft_label


def prob_to_corr_test(prob_matrix):
    c = torch.zeros_like(input=prob_matrix)
    idx = torch.argmax(input=prob_matrix, dim=2, keepdim=True)
    for bsize in range(c.shape[0]):
        for each_row in range(c.shape[1]):
            c[bsize][each_row][idx[bsize][each_row]] = 1.0

    return c


def label_ACC_percentage_for_inference(label_in, label_gt):
    assert (label_in.shape == label_gt.shape)
    bsize = label_in.shape[0]
    b_acc = []
    for i in range(bsize):
        element_product = torch.mul(label_in[i], label_gt[i])
        N1 = label_in[i].shape[0]
        sum_row = torch.sum(element_product, dim=-1)  # N1x1

        hit = (sum_row != 0).sum()
        acc = hit.float() / torch.tensor(N1).float()
        b_acc.append(acc * 100.0)
    mean = torch.mean(torch.stack(b_acc))
    return mean


# with SHREC corrcted softlabel
def test_with_tolerance_SHREC(net, test_loader, name=None):
    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'resume'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    if name is not None:
        print('Loading specific from {}'.format(name))
        load_model(net, name)
    else:
        model_path = str(checkpoint_path / 'params_best.pt')
        print('Loading best model parameters from {}'.format(model_path))
        load_model(net, model_path)

    net.eval()
    total_acc = 0
    tolerance = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    acc_with_tolerance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_examples = 0
    item = 0
    batch_size = cfg.DATASET.BATCH_SIZE

    for corr, src, tgt, sl in tqdm(test_loader):
        num_examples += 1
        src = src.cuda()
        tgt = tgt.cuda()
        label = corr.cuda()
        batch_size = src.shape[0]
        num_points = src.shape[1]
        p = net(src, tgt)
        corr_tensor = prob_to_corr_test(p)
        acc_000 = label_ACC_percentage_for_inference(corr_tensor, label)
        acc_with_tolerance[0] = acc_with_tolerance[0] + acc_000
        print('---Test---BCE tolerance: 0, ACC: %f' % acc_000)

        for idx, sl_now in enumerate(sl):
            sl_now = sl_now.cuda()
            acc_now = label_ACC_percentage_for_inference(corr_tensor, sl_now)
            acc_with_tolerance[idx+1] = acc_with_tolerance[idx+1] + acc_now
            print('---Test---BCE tolerance: %f, ACC: %f' % (tolerance[idx], acc_now))


    print('tolerance_0:', (acc_with_tolerance[0] / num_examples).item())
    for idx, acc in enumerate(tolerance):
        print('tolerance_: %f, ACC: %f' % (tolerance[idx], (acc_with_tolerance[idx+1] / num_examples).item()))


def test_notolerance(net, test_loader, name=None):
    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'resume'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    if name is not None:
        print('Loading specific from {}'.format(name))
        load_model(net, name)
    else:
        model_path = str(checkpoint_path / 'params_best.pt')
        print('Loading best model parameters from {}'.format(model_path))
        load_model(net, model_path)

    net.eval()
    total_acc = 0
    num_examples = 0

    for corr, src, tgt in tqdm(test_loader):
        num_examples += 1
        src = src.cuda()
        tgt = tgt.cuda()
        corr = corr.cuda()

        p = net(src, tgt)
        corr_tensor = prob_to_corr_test(p)
        acc_000 = label_ACC_percentage_for_inference(corr_tensor, corr)
        print('Accuracy_tolerance_0:', acc_000.item())
        total_acc += acc_000.item()


    t_acc = total_acc/num_examples
    print('---Test TOTAL--- ACC: %f' % t_acc)


if __name__ == '__main__':
    from utils.parse_argspc import parse_args

    TIMESTAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args = parse_args('Non-rigid registration of graph matching training & evaluation code.')

    net = Net.Net()
    net.cuda()

    test_loader = DataLoader(SHRECTest(), batch_size=cfg.DATASET.BATCH_SIZE)
    test_loader_SHRECTest_witout = DataLoader(SHRECTest_witout(), batch_size=cfg.DATASET.BATCH_SIZE)

    if cfg.TRAIN.OPTIM == "SGD":
        opt = optim.SGD(net.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-4, nesterov=True)
    if cfg.TRAIN.OPTIM == "Adam":
        opt = optim.Adam(net.parameters())

    test_with_tolerance_SHREC(net, test_loader, "pretrained.pt")

    test_notolerance(net, test_loader_SHRECTest_witout, "pretrained.pt")


