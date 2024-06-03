#!.env/bin/python
from logging import config
import os
import cv2
import glob
import copy
import time
import wandb
import random
import argparse
import datetime
import rasterio

import numpy as np
from compress_BigEarthTT import MACHINE_ALIAS
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
MACHINE_ALIAS = 'LH'
METRICS_FILE = "BigEarthNet_comparison_metrics_"

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

def initalize_wandb(args, timestamp, tags = []):
    run_config = wandb.config = {
        "seed_idx": args.seed_idx,
        "epsilon": args.epsilon,
        "reshaping": args.reshaping,
        "batch_size": args.batch_size,
        "numpy": args.numpy,
        "wandb": args.wandb,
        "method": args.method
    }
    wandb.init(
        project="HierarchicalTucker_experiments",
        name = f"BigEarthNet_{args.method}_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
        config = run_config,
        tags = tags,
    )

def hierarchical_tucker_bigearth(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if args.numpy:
        image_files = glob.glob(DIRECTORIES["data"]+"*.npy")
    else:
        raise NotImplementedError("Only the numpy case is implemented. Exiting...")
    random.shuffle(image_files)
    num_images = len(image_files)
    train_images = image_files[:int(num_images * TRAIN_RATIO)]
    image_folders = image_files[int(num_images * TRAIN_RATIO):]
    val_images = image_folders[:int(num_images * VAL_RATIO)]
    image_folders = image_folders[int(num_images * VAL_RATIO):]
    test_images = image_folders[:int(num_images * TEST_RATIO)]
    print("Number of train images: ", len(train_images))
    print("Number of val images: ", len(val_images))
    print("Number of test images: ", len(test_images))

    total_time = 0
    train_batch = []
    compressed_batch = []
    for _ in range(args.batch_size):
        image_dir = train_images.pop(0)
        compressed_batch.append(image_dir)
        train_data = np.load(image_dir).reshape(args.reshaping, order=ORD)[...,None]
        train_batch.append(train_data)
    train_batch = np.concatenate(train_batch, axis=-1)
    print(train_batch.shape)
    batch_norm = nla.norm(train_batch)
    batch_along = len(train_batch.shape)-1

    dataset = ht.HTucker()
    dataset.initialize(
        train_batch,
        # batch = True,
        # batch_dimension= batch_along,
    )
    print(dataset.original_shape)
    dimension_tree = ht.createDimensionTree(
        dataset.original_shape,
        numSplits = 2,
        minSplitSize = 1,
    )
    dimension_tree.get_items_from_level()
    dataset.rtol = args.epsilon
    tic = time.time()
    dataset.compress_leaf2root(
        train_batch,
        dimension_tree,
    )
    compression_time = time.time() - tic

    # deneme  = dataset
    # deneme.leaves[-1].core=deneme.leaves[-1].core[0,:][None,...]

    # print()
    # print(deneme.root.shape)
    # for core in deneme.transfer_nodes:
    #     print(core.shape)
    # for core in deneme.leaves:
    #     print(core.shape)
    # print()

    # deneme.reconstruct_all()
    # print(deneme.root.shape)
    print()

    metrics = []
    for image_idx, image_dir in enumerate(compressed_batch):
        temp_ht  = copy.deepcopy(dataset)
        temp_ht.leaves[-1].core = temp_ht.leaves[-1].core[image_idx,:][None,...]
        temp_ht.reconstruct_all()
        im_norm = nla.norm(train_batch[...,image_idx])
        error = nla.norm(train_batch[...,image_idx][...,None]-temp_ht.root.core)/im_norm
        metrics.append([error, im_norm, nla.norm(temp_ht.root.core)])
        # print(error, im_norm, nla.norm(temp_ht.root.core))
    # print(np.mean(errors))
    metrics=np.array(metrics)
    # print(np.array(metrics).shape)
    # print(np.array(metrics))

    print()
    print(compression_time, dataset.compression_ratio, np.mean(metrics[:,0]),  np.sqrt(1-(np.square(metrics[:,2]).sum()/np.square(metrics[:,1]).sum())))
    print()
    
    
    ranks =[list(dataset.root.shape)]
    for core in dataset.transfer_nodes:
        ranks.append(list(core.shape))
    for leaf in dataset.leaves:
        ranks.append(list(leaf.shape))
    print(ranks)
    print(dataset.original_shape)
        
    line_to_append = []
    line_to_append.append(f"{args.batch_size:06d}")
    line_to_append.append(f"{dataset.root.core.shape[0]:06d}")
    line_to_append.append(f"{dataset.root.core.shape[1]:06d}")
    line_to_append.append(f"{np.prod(dataset.original_shape)/(dataset.root.core.shape[1]*dataset.root.core.shape[1]*args.batch_size):010.5f}")
    line_to_append.append(f"{round(dataset.compression_ratio, 5):010.5f}")
    line_to_append.append(f"{round(compression_time, 5):08.5f}")
    line_to_append.append(f"{round(np.mean(metrics[:,0]), 5):0.5f}")
    line_to_append.append(f"{round(np.sqrt(1-(np.square(metrics[:,2]).sum()/np.square(metrics[:,1]).sum())),5):0.5f}")
    line_to_append.append(f"{batch_norm:14.5f}")
    line_to_append.append(" ".join(map(str,ranks)))
    # line_to_append.append(f"{' '.join(map(lambda x: f'{x:03d}', ranks))}")
    print(" ".join(line_to_append))
    with open(DIRECTORIES["metric"]+METRICS_FILE+f"eps_{args.epsilon}_{args.method}.txt", 'a') as f:
        f.writelines(" ".join(map(str,line_to_append))+"\n")



    # line_to_append.append(f"{round(total_time, 5):09.5f}")
    # print(f"{args.batch_size:06d} {dataset.root.core.shape[-1]:06d} {round(compression_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
    
    # print(dataset.root.shape)
    # for core in dataset.transfer_nodes:
    #     print(core.shape)
    # for core in dataset.leaves:
    #     print(core.shape)
    # print()

    # Load the data

    # Compress the tensor

    # Test reconstruction
    
    # Test generalization
    # raise NotImplementedError

def batch_hierarchical_tucker_bigearth(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raise NotImplementedError

if __name__ == "__main__":
    overall_start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    args = parser_bigearthnet()

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
        args.seed_idx = seed_idx
    else:
        seed_idx = args.seed_idx

    random.seed(seed_idx)  # Fix the seed to a random number
    np.random.seed(seed_idx)  # Fix the seed to a random number

    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for BigEarthNet")
        args.reshaping = [12,10,12,10,12]
    print("Reshaping used: ", args.reshaping)

    if args.data_location is None:
        if args.numpy:
            if os.path.exists(reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data","BigEarthNet-v1.0_numpy"])):
                print("Data location is not provided, using default location")
                data_loc = reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data","BigEarthNet-v1.0_numpy"])
            else:
                raise IsADirectoryError("Please provide the data location")
        else:
            if os.path.exists(reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data","BigEarthNet-v1.0"])):
                print("Data location is not provided, using default location")
                data_loc = reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data","BigEarthNet-v1.0"])
            else:
                raise IsADirectoryError("Please provide the data location")
    else: 
        data_loc = args.data_location
        if args.numpy:
            assert glob.glob(args.data_location+"*.npy"), "There are no numpy files in the provided directory."
    
    DIRECTORIES = {
        "data" : data_loc+PATH_SEP,
        "cores" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["savedCores"])+PATH_SEP,
        "metric" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["experiments","BigEarthNet"])+PATH_SEP,
    }

    if not os.path.exists(DIRECTORIES["metric"]):
        os.makedirs(DIRECTORIES["metric"])
        
    if args.wandb:
        initalize_wandb(args, timestamp)

    print(args)
    # print(len(glob.glob(data_loc+"*.npy")))

    if args.method == "ht":
        hierarchical_tucker_bigearth(args)
    elif args.method == "bht":
        batch_hierarchical_tucker_bigearth(args)
    else:
        print(f"Method should be either ht or bht, not '{args.method}'. Exiting...")

    overall_time = time.time() - overall_start