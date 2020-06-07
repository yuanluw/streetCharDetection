# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 15:56, matt '

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from model.base import base_model
from dataset.normal_dataset import SVHNDataset, test_transform

import config


def get_test_loader(df_submit, batch_size):
    images_name = list(df_submit['file_name'])
    images_path = [os.path.join(config.dataset_path, "mchar_test_a", x) for x in images_name]
    normal_dataset = SVHNDataset(images_path, None, test_transform)
    dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=config.num_workers)
    return dataloader


def run(arg):
    # test_data = get_normal_dataset(arg.test_batch_size, index="test")

    df_submit = pd.read_csv("mchar_sample_submit_A.csv")
    test_data = get_test_loader(df_submit, arg.test_batch_size)

    if arg.net == "base_model":
        model = base_model()

    model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, arg.dataset_name + "_" + arg.net +
                                                  "_" + arg.checkpoint + '.pth')))

    model.to(config.device)
    model.eval()

    test_pred = []
    process_bar = tqdm(total=len(test_data))
    for i, (img) in enumerate(test_data):
        process_bar.update(1)
        with torch.no_grad():
            img = img.to(config.device)
            c1, c2, c3, c4, c5, c6 = model(img)

        output = np.concatenate([c1.data.cpu().numpy(),
                                 c2.data.cpu().numpy(),
                                 c3.data.cpu().numpy(),
                                 c4.data.cpu().numpy(),
                                 c5.data.cpu().numpy(),
                                 c6.data.cpu().numpy(),
                                 ], axis=1)

        test_pred.append(output)
    test_pred = np.vstack(test_pred)
    predict_label = np.vstack([
        test_pred[:, :11].argmax(1),
        test_pred[:, 11:22].argmax(1),
        test_pred[:, 22:33].argmax(1),
        test_pred[:, 33:44].argmax(1),
        test_pred[:, 44:55].argmax(1),
        test_pred[:, 55:66].argmax(1),
    ]).T

    val_label_pred = []
    for x in predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))

    print(len(val_label_pred))

    df_submit['file_code'] = val_label_pred
    df_submit.to_csv('submit.csv', index=None)

