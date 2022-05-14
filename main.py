import argparse

import torch
import torch.optim as optim

from lena.engine.train import get_cifar10, train
from lena.model.resnet import senet50
from lena.optim import build_optimizer


def main(args):
    train_loader, val_loader = get_cifar10(args.root, args.batch)
    net = senet50(3, 10)
    optimizer = build_optimizer(args.optim, net.parameters(), lr=args.lr)

    train(train_loader, val_loader, args.epoch, net, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim', type=str, required=True, choices=['Adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--root', type=str, default='/tmp')
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()
    main(args)
