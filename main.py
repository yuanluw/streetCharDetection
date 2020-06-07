# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/6/7 13:47, matt '


import argparse


def get_augments():
    parser = argparse.ArgumentParser(description="pytorch street char detection")

    parser.add_argument("--action", type=str, default="train", choices=("train", "test"))
    parser.add_argument("--mul_gpu", type=int, default=0, help="use multiple gpu(default: 0")
    parser.add_argument("--net", type=str, default="base_model", choices=("base_model", "st_model"))
    parser.add_argument("--dataset_name", default="normal", choices=("normal",))

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate(default: 0.1)")
    parser.add_argument("-lr_step", type=int, default=10, help="period of learning rate decay")
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam"), help='optimizer')
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")


    return parser.parse_args()


def main():
    arg = get_augments()
    if arg.action == "train":
        from train import run
        run(arg)
    elif arg.action == "test":
        from test import run
        run(arg)


if __name__ == "__main__":
    main()

