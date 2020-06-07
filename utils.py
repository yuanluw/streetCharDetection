# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 13:07, matt '


import os
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import config


def count_time(prev_time, cur_time):
    h, reminder = divmod((cur_time-prev_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = "time %02d:%02d:%02d" %(h, m, s)
    return time_str


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(dataset_name, net_name, epoch, feature_net):

    feature_state_dict = feature_net.state_dict()
    torch.save(feature_state_dict, os.path.join(config.checkpoint_path, dataset_name + "_" + net_name + "_" +
                                                str(epoch) + '.pth'))



