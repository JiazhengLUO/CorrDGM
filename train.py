import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import SurrealTrain, SHRECTest_witout
from models import Net
from tensorboardX import SummaryWriter
from datetime import datetime
from utils.config import cfg
from pathlib import Path
from utils.model_sl import load_model, save_model
from utils.loss_func import FocalLoss
from typing import Optional
from shutil import copy
import os
import scipy.optimize as opt


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


def test_one_epoch3(net, test_loader, logger):
    net.eval()
    total_acc = 0
    num_examples = 0

    if cfg.TRAIN.LOSS_FUNC == 'BCELoss':
        loss = torch.nn.BCELoss()
    if cfg.TRAIN.LOSS_FUNC == 'Focal':
        loss = FocalLoss()

    for corr, src, tgt in tqdm(test_loader):
        num_examples += 1
        pinput1 = src.cuda()
        input2 = tgt.cuda()
        label = corr.cuda()
        with torch.no_grad():
            p = net(pinput1, input2)
            output = loss(p, label)
            corr_tensor = prob_to_corr_test(p)
            acc_000 = label_ACC_percentage_for_inference(corr_tensor, label)
            total_acc += acc_000.item()

        tqdm.write('---Test--- ACC: %f' % acc_000.item())

    t_acc = total_acc/num_examples
    tqdm.write('---Test AVE--- ACC: %f' % t_acc)
    print('---Test AVE--- ACC: %f' % t_acc, file=logger, flush=True)

    return t_acc, output


def train_one_epoch(net, train_loader, opt, logger):
    net.train()
    total_acc = 0
    num_examples = 0

    loss_type = cfg.TRAIN.LOSS_FUNC

    if loss_type == 'BCELoss':
        loss = torch.nn.BCELoss()
    if loss_type == 'Focal':
        loss = FocalLoss()
    # use_focal_loss = False

    for corr, src, tgt in tqdm(train_loader):
        num_examples += 1
        src = src.cuda()
        tgt = tgt.cuda()
        corr = corr.cuda()

        opt.zero_grad()

        corr_est = net(src, tgt)

        output = loss(corr_est, corr)

        output.backward()

        opt.step()

        corr_est_tensor = prob_to_corr_test(corr_est)
        acc = label_ACC_percentage_for_inference(corr_est_tensor, corr)

        # print('---Train--- ACC: %f, BCE_loss: %f' % (acc.item(), output))
        tqdm.write('---Train--- ACC: %f, %s : %f' % (acc.item(), loss_type, output))

        total_acc = total_acc + acc.item()

    t_acc = total_acc/num_examples
    print('---Train AVE--- ACC: %f' % t_acc, file=logger, flush=True)
    tqdm.write('---Train AVE--- ACC: %f' % t_acc)

    return t_acc, output


def train(net, train_loader, test_loader, opt, num_epochs, resume, start_epoch, savefiletime):

    checkpoint_Path = cfg.OUTPUT_PATH + '/' + str(savefiletime) + '/params'
    if not Path(checkpoint_Path).exists():
        Path(checkpoint_Path).mkdir(parents=True)

    resume_Path = cfg.OUTPUT_PATH + '/resume'
    if not Path(resume_Path).exists():
        Path(resume_Path).mkdir(parents=True)

    logger_Path = cfg.OUTPUT_PATH + '/' + str(savefiletime)
    if not Path(logger_Path).exists():
        Path(logger_Path).mkdir(parents=True)

    logger = open(logger_Path + '/log_{}.txt'.format(savefiletime), 'w')

    writer = SummaryWriter('runs/'+savefiletime)

    if resume:
        assert start_epoch != 0
        model_path = resume_Path + '/params_{:04}.pt'.format(start_epoch)
        print('Loading model parameters from {}'.format(model_path))
        load_model(net, model_path)

        optim_path = resume_Path + '/optim_{:04}.pt'.format(start_epoch)
        print('Loading optimizer state from {}'.format(optim_path))
        opt.load_state_dict(torch.load(optim_path))

    for epoch in range(num_epochs):
        print("epoch: %d" % epoch, file=logger, flush=True)
        tqdm.write("epoch: %d" % epoch)
        # train
        acc_train, loss_train = train_one_epoch(net, train_loader, opt, logger)
        writer.add_scalar('train/loss', loss_train, global_step=epoch)
        writer.add_scalar('train/acc', acc_train, global_step=epoch)

        acc_test, loss_test = test_one_epoch3(net, test_loader, logger)
        writer.add_scalar('test/loss', loss_test, global_step=epoch)
        writer.add_scalar('test/acc', acc_test, global_step=epoch)

        # if epoch % 5 == 0:
        save_model(net, checkpoint_Path + '/params_{:04}.pt'.format(epoch + 1))
        torch.save(opt.state_dict(), checkpoint_Path + '/optim_{:04}.pt'.format(epoch + 1))

    writer.close()


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    import socket

    TIMESTAMP = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args = parse_args('training.')

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    cfglogs_Path = cfg.OUTPUT_PATH + '/' + str(TIMESTAMP)
    if not Path(cfglogs_Path).exists():
        Path(cfglogs_Path).mkdir(parents=True)
    with DupStdoutFileManager(str(Path(cfglogs_Path) / ('config_' + TIMESTAMP + '.log'))) as _:
        print_easydict(cfg)

    copy('models/Net.py', cfglogs_Path)
    copy('train.py', cfglogs_Path)

    # discription for this trian/test try
    discirplog = open(cfglogs_Path + ('/discrip_' + TIMESTAMP + '.txt'), 'w')
    print('Train', file=discirplog, flush=True)

    net = Net.Net()
    net.cuda()

    train_loader = DataLoader(SurrealTrain(size=cfg.DATASET.USE_SIZE), batch_size=cfg.DATASET.BATCH_SIZE, drop_last=True)
    test_loader_SHRECTest_witout = DataLoader(SHRECTest_witout(), batch_size=cfg.DATASET.BATCH_SIZE)

    if cfg.TRAIN.OPTIM == "SGD":
        opt = optim.SGD(net.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-4, nesterov=True)
    if cfg.TRAIN.OPTIM == "Adam":
        opt = optim.Adam(net.parameters())

    train(net, train_loader, test_loader_SHRECTest_witout, opt,
          num_epochs=cfg.TRAIN.NUM_EPOCHS,
          resume=cfg.TRAIN.START_EPOCH != 0,
          start_epoch=cfg.TRAIN.START_EPOCH,
          savefiletime=TIMESTAMP
    )

    print('FINISH')


