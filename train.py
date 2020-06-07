# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 13:07, matt '


import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from dataset.normal_dataset import get_normal_dataset
from model.base import base_model, st_model
import config
from utils import count_time, get_logger, AverageMeter, save_checkpoint

cur_path = os.path.abspath(os.path.dirname(__file__))


def run(arg):
    torch.manual_seed(7)
    np.random.seed(7)
    print("lr %f, epoch_num %d, decay_rate %f gamma %f" % (arg.lr, arg.epochs, arg.decay, arg.gamma))

    start_epoch = 0

    train_data = get_normal_dataset(arg.train_batch_size, index="train")
    val_data = get_normal_dataset(arg.train_batch_size, index="val")

    if arg.net == "base_model":
        model = base_model()
    elif arg.net == "st_model":
        model = st_model()

    if arg.checkpoint is not None:
        print("load pre train model")
        model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, arg.dataset_name + "_" + arg.net +
                                                            "_" +arg.checkpoint+'.pth')))

    if arg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=arg.lr, momentum=arg.momentum, weight_decay=arg.decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=arg.lr_step, gamma=0.1)
    elif arg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.decay)

    model = model.to(config.device)

    if arg.mul_gpu:
        model = nn.DataParallel(model)

    logger = get_logger()

    criterion = nn.CrossEntropyLoss()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("start training: ", datetime.now())

    best_acc = 0.0
    for epoch in range(start_epoch, arg.epochs):
        prev_time = datetime.now()
        train_loss = train(train_data, model, criterion, optimizer, epoch, logger)
        val_loss = validate(val_data, model, criterion)
        val_acc = predict(val_data, model)
        now_time = datetime.now()
        time_str = count_time(prev_time, now_time)
        print("train: current (%d/%d) batch train loss is %f val loss is %f val acc is %f time "
              "is %s" % (epoch, arg.epochs, train_loss, val_loss, val_acc, time_str))
        if arg.optimizer == "sgd":
            scheduler.step()

        if best_acc < val_acc:
            save_checkpoint(arg.dataset_name, arg.net, 0, model)

    save_checkpoint(arg.dataset_name, arg.net, arg.epochs, model)


def train(train_data, model, criterion, optimizer, epoch, logger):
    model.train()
    losses = AverageMeter()

    for i, (img, label) in enumerate(train_data):
        img = img.to(config.device)
        label = label.to(config.device)

        c1, c2, c3, c4, c5, c6 = model(img)

        # calculate loss
        loss = criterion(c1, label[:, 0]) + criterion(c2, label[:, 1]) + criterion(c3, label[:, 2]) +\
               criterion(c4, label[:, 3]) + criterion(c5, label[:, 4]) + criterion(c6, label[:, 5])

        # bp
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # metrics track
        losses.update(loss.item())

        if i % config.print_freq == 0:
            logger.info('Epoch: [{0}][{1}][{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        .format(epoch, i, len(train_data), loss=losses))

    return losses.avg


def validate(val_data, model, criterion):
    model.eval()
    losses = AverageMeter()

    for i, (img, label) in enumerate(val_data):
        img = img.to(config.device)
        label = label.to(config.device)

        c1, c2, c3, c4, c5, c6 = model(img)

        # calculate loss
        loss = criterion(c1, label[:, 0]) + criterion(c2, label[:, 1]) + criterion(c3, label[:, 2]) +\
               criterion(c4, label[:, 3]) + criterion(c5, label[:, 4]) + criterion(c6, label[:, 5])

        # metrics track
        losses.update(loss.item())

    return losses.avg


def predict(val_data, model):
    model.eval()
    acc = AverageMeter()
    test_pred = []
    for i, (img, label) in enumerate(val_data):
        img = img.to(config.device)

        c1, c2, c3, c4, c5, c6 = model(img)
        output = np.concatenate([c1.data.cpu().numpy(),
                                 c2.data.cpu().numpy(),
                                 c3.data.cpu().numpy(),
                                 c4.data.cpu().numpy(),
                                 c5.data.cpu().numpy(),
                                 c6.data.cpu().numpy(),
                                 ], axis=1)

        predict_label = np.vstack([
            output[:, :11].argmax(1),
            output[:, 11:22].argmax(1),
            output[:, 22:33].argmax(1),
            output[:, 33:44].argmax(1),
            output[:, 44:55].argmax(1),
            output[:, 55:66].argmax(1),
        ]).T

        val_label_pred = []
        for x in predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_label = [''.join(map(str, x[x != 10])) for x in np.array(label)]
        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
        acc.update(val_char_acc)

    return acc.avg