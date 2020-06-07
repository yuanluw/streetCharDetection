# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 12:18, matt '

import torch

dataset_path = '/media/wyl/streetCharDataset/'
num_workers = 8
checkpoint_path = "/home/wyl/codeFile/streetCharDetection/pre_train/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_freq = 100