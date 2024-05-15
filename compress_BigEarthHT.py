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
import numpy.linalg as nla

import htucker as ht
from PIL import Image
from multiprocessing import Pool
from functools import reduce, partial


MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"

__all__ = ["read_image"]

def read_image(band_name, image_path, resizing = None, interpolation_method = cv2.INTER_AREA):
    image_name = image_path.split(PATH_SEP)[-1]
    im = rasterio.open(image_path+PATH_SEP+image_name+f"_{band_name}.tif").read(1)
    # im = np.array(Image.open(image_path+PATH_SEP+image_name+f"_{band_name}.tif"))
    # im = cv2.imread(image_path+PATH_SEP+image_name+f"_{band_name}.tif", cv2.IMREAD_UNCHANGED)
    if resizing:
        im = cv2.resize(im, resizing, interpolation=interpolation_method)
    return im


def main():
    parser = argparse.ArgumentParser(description='This script reads the BigEarthNet image patches')
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', help='epsilon value', default=0.1)
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Read the images from numpy files', default=False)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)

    args = parser.parse_args()

    filter = [False, True, True, True, True, True, True, True, True, False, True, True]
    filter = [True, True, True, True, True, True, True, True, True, True, True, True]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # print("Timestamp: ", timestamp)
    # print("BigEarthNet_epsilon_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp)
    # return 0
    if args.wandb:
        wandb.init(
            # entity="dorukaks",
            project="HierarchicalTucker_experiments",
            name="BigEarthNet_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
        )
        wandb.config = {
            "epsilon": args.epsilon,
            "batch_size": args.batch_size,
            "reshaping": args.reshaping,
            "numpy": args.numpy,
            "filter": filter,
            "seed_idx": args.seed_idx,
        }
        wandb.init(
            # entity="dorukaks",
            project="HierarchicalTucker_experiments",
            name="BigEarthNet_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
            config=run_config,
        )


    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
    else:
        seed_idx = args.seed_idx

    print("Seed index is: ", seed_idx)
    random.seed(seed_idx)  # Fix the seed to a random number
    np.random.seed(seed_idx)  # Fix the seed to a random number
    print(args)
    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for BigEarthNet")
        args.reshaping = [12,10,12,10,12]
    args.reshaping[-1] = sum(filter) 
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
    

    directories = {
        "data" : data_loc+PATH_SEP,
        "cores" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["savedCores"])+PATH_SEP,
        "metric" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["experiments","BigEarthNet"])+PATH_SEP,

    }

    if not os.path.exists(directories["cores"]):
        os.makedirs(directories["cores"])
    if not os.path.exists(directories["metric"]):
        os.makedirs(directories["metric"])
        
        
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
                'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    image_folders = glob.glob(directories["data"]+"*")
    num_images = len(image_folders)
    print("Number of images: ", num_images)
    random.shuffle(image_folders)

    train_images = image_folders[:int(num_images * TRAIN_RATIO)]
    image_folders = image_folders[int(num_images * TRAIN_RATIO):]
    val_images = image_folders[:int(num_images * VAL_RATIO)]
    image_folders = image_folders[int(num_images * VAL_RATIO):]
    test_images = image_folders[:int(num_images * TEST_RATIO)]
    print("Number of train images: ", len(train_images))
    print("Number of val images: ", len(val_images))
    print("Number of test images: ", len(test_images))
    total_time = 0
    batch_index = 0
    if args.numpy:
        train_batch = []
        tic = time.time()
        for _ in range(args.batch_size):
            image_dir = train_images.pop(0)
            train_data = np.load(image_dir)[...,filter].reshape(args.reshaping)[...,None]
            train_batch.append(train_data)
        toc = time.time()-tic
        train_batch = np.concatenate(train_batch, axis=-1)

        batch_along = len(train_batch.shape)-1
        
        dataset = ht.HTucker()
        dataset.initialize(
            train_batch,
            batch=True,
            batch_dimension=batch_along,
        )
        dimenion_tree = ht.createDimensionTree(
            dataset.original_shape[:batch_along]+dataset.original_shape[batch_along+1:],
            numSplits = 2,
            minSplitSize = 1,
        )
        dimenion_tree.get_items_from_level()
        dataset.rtol = args.epsilon
        tic = time.time()
        dataset.compress_leaf2root_batch(
            train_batch,
            dimension_tree=dimenion_tree,
            batch_dimension=batch_along,
        )
        batch_time = time.time()-tic
        total_time += batch_time

        rec = dataset.reconstruct(
            dataset.root.core[...,-args.batch_size:]
        )
        error_before_update = 0
        error_after_update = nla.norm(rec-train_batch)/nla.norm(train_batch)
        # train_images = train_images[:100*args.batch_size]
        val_errors = []
        for image in val_images:
            val_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
            rec = dataset.reconstruct(
                dataset.project(
                    val_data,
                    batch=True,
                    batch_dimension=batch_along
                )
            )
            approximation_error = nla.norm(rec-val_data)/nla.norm(val_data)
            val_errors.append(approximation_error)
            # print(approximation_error)
        test_errors = []
        for image in test_images:
            test_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
            rec = dataset.reconstruct(
                dataset.project(
                    test_data,
                    batch=True,
                    batch_dimension=batch_along
                )
            )
            approximation_error = nla.norm(rec-test_data)/nla.norm(test_data)
            test_errors.append(approximation_error)
        ranks = []
        for core in dataset.transfer_nodes:
            ranks.append(core.shape[-1])
        for leaf in dataset.leaves:
            ranks.append(leaf.shape[-1])
        print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
        logging_dict = {
                "compression_ratio": dataset.compression_ratio,
                "error_before_update": error_before_update, 
                "error_after_update": error_after_update,
                "image_count": dataset.root.core.shape[-1],
                "batch_time": batch_time,
                "total_time": total_time,
                "val_error": np.mean(val_errors),
                "test_error": np.mean(test_errors),
        }
        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank

        if args.wandb:
            wandb.log(logging_dict)

        while train_images:
            batch_index += 1
            train_batch = []
            for _ in range(args.batch_size):
                image_dir = train_images.pop(0)
                train_data = np.load(image_dir)[...,filter].reshape(args.reshaping)[...,None]
                train_batch.append(train_data)
            train_batch = np.concatenate(train_batch, axis=-1)
            projection = dataset.reconstruct(
                dataset.project(
                train_batch,
                batch=True,
                batch_dimension=batch_along
            ))
            error_before_update = nla.norm(projection-train_batch)/nla.norm(train_batch)
            tic = time.time()
            update_flag = dataset.incremental_update_batch(
                train_batch,
                batch_dimension=batch_along,
                append=True,
            )
            batch_time = time.time()-tic
            rec = dataset.reconstruct(
                dataset.root.core[...,-args.batch_size:]
            )
            error_after_update = nla.norm(rec-train_batch)/nla.norm(train_batch)
            total_time += batch_time
            if update_flag:
                val_errors = []
                for image in val_images:
                    val_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
                    rec = dataset.reconstruct(
                        dataset.project(
                            val_data,
                            batch=True,
                            batch_dimension=batch_along
                        )
                    )
                    approximation_error = nla.norm(rec-val_data)/nla.norm(val_data)
                    val_errors.append(approximation_error)
                test_errors = []
                for image in test_images:
                    test_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
                    rec = dataset.reconstruct(
                        dataset.project(
                            test_data,
                            batch=True,
                            batch_dimension=batch_along
                        )
                    )
                    approximation_error = nla.norm(rec-test_data)/nla.norm(test_data)
                    test_errors.append(approximation_error)
                ranks = []
                for core in dataset.transfer_nodes:
                    ranks.append(core.shape[-1])
                for leaf in dataset.leaves:
                    ranks.append(leaf.shape[-1])
            
            logging_dict = {
                "compression_ratio": dataset.compression_ratio,
                "error_before_update": error_before_update, 
                "error_after_update": error_after_update,
                "image_count": dataset.root.core.shape[-1],
                "batch_time": batch_time,
                "total_time": total_time,
                "val_error": np.mean(val_errors),
                "test_error": np.mean(test_errors),
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            if args.wandb:
                wandb.log(logging_dict)

            print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
            # {' '.join(map(lambda x: f'{x:03d}', ranks))}

        numel = np.prod(dataset.root.core.shape)
        print(dataset.root.shape)
        for core in dataset.transfer_nodes:
            print(core.shape)
            numel += np.prod(core.shape)
        for leaf in dataset.leaves:
            print(leaf.shape)
            numel += np.prod(leaf.shape)
        print(numel)
        print(np.prod(train_batch.shape[:-1]),dataset.root.shape[-1])
        print(dataset.batch_count)
        print(dataset.original_shape)

    else:
        image_size = cv2.imread(train_images[0]+PATH_SEP+os.listdir(train_images[0])[0], cv2.IMREAD_UNCHANGED).shape

        for band in band_names:
            f = rasterio.open(train_images[0]+PATH_SEP+train_images[0].split(PATH_SEP)[-1]+f"_{band}.tif")
            if np.prod(f.read(1).shape) > np.prod(image_size):
                image_size = f.read(1).shape
            f.close()
        print("Maximum image size: ", image_size)
        assert np.prod(args.reshaping) == np.prod(image_size)*len(band_names), "Reshaping size does not match the image size"
        print("Reshaping used: ", args.reshaping)

        
        # Create the dataset object and compute first set of cores
        def load_images(image_path):
            images = []
            for band in band_names:
                image = read_image(image_path, band, resizing=args.reshaping)
                images.append(image)
            return np.array(images)
        
        # for aaa in zip([train_images[0]]*len(band_names), band_names, [args.reshaping]*len(band_names)):
        #     print(aaa)
        # return 0
        # print(train_images[:args.batch_size])
        train_batch = []
        tic = time.time()
        for _ in range(args.batch_size):
            image_dir = train_images.pop(0)
            # print(image_dir)
            with Pool() as pool:
                train_data = pool.map(partial(read_image,image_path=image_dir,resizing=image_size,interpolation_method=cv2.INTER_AREA),band_names,)
            train_data = np.array(train_data).transpose(1,2,0).reshape(args.reshaping, order=ORD)[...,None]
            train_batch.append(train_data)
        toc = time.time()-tic
        print(toc)
        train_batch = np.concatenate(train_batch, axis=-1)
        print(train_batch.shape)

    if args.wandb:
        wandb.finish()
    


if __name__ == '__main__':
    main()