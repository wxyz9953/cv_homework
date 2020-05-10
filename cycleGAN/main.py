import os
from argparse import ArgumentParser
from model import cycleGAN
from utils import create_link
from test import test


def get_args():
    parser = ArgumentParser(description='CycleGAN')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--load_height', type=int, default=286)
    parser.add_argument('--load_width', type=int, default=286)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/monet2photo')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/monet2photo')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    create_link(args.dataset_dir)

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    if args.training:
        print("Training")
        model = cycleGAN(args)
        model.train(args)
    if args.testing:
        print("Testing")
        test(args)


if __name__ == '__main__':
    main()
