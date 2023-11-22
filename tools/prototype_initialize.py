import argparse

import mmcv
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Target Prototype and initialize"
    )
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
