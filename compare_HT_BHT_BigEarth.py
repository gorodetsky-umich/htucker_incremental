#!.env/bin/python
import os
import cv2
import glob
import time
import wandb
import random
import argparse
import datetime
import rasterio

import numpy as np
import htucker as ht
import numpy.linalg as nla

from PIL import Image
from multiprocessing import Pool
from functools import partial, reduce

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"

def parser_bigearthnet():
    parser = argparse.ArgumentParser(description="BigEarthNet")
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, help='epsilon value', default=0.1)
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Read the images from numpy files', default=False)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    parser.add_argument('-m', '--method', dest='method', help='method of compression', default=None)
    args = parser.parse_args()
    return args


def hierarchical_tucker_bigearth(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    raise NotImplementedError

def batch_hierarchical_tucker_bigearth(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raise NotImplementedError

if __name__ == "__main__":
    overall_start = time.time()

    args = parser_bigearthnet()

    if args.method == "ht":
        hierarchical_tucker_bigearth(args)
    elif args.method == "bht":
        batch_hierarchical_tucker_bigearth(args)
    else:
        print(f"Method should be either ht or bht, not '{args.method}'. Exiting...")

    overall_time = time.time() - overall_start