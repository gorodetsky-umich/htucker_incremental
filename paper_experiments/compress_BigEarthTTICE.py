#!.env/bin/python
import argparse
import datetime
import glob
import os
import random
import sys
import time

import numpy as np

import wandb

sys.path.insert(0, "/home/doruk/TT-ICE/")
from functools import reduce

import DaMAT as dmt
import numpy.linalg as nla

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.96
VAL_RATIO = 0.02
TEST_RATIO = 0.02
ORD = "F"
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
HEUR_2_USE = ["skip", "occupancy"]
MACHINE_ALIAS = "LH"
OCCUPANCY = 1


def get_args():
    parser = argparse.ArgumentParser(description="This script reads the BigEarthNet image patches")
    parser.add_argument(
        "-d", "--data_location", dest="data_location", help="path to data", default=None
    )
    parser.add_argument(
        "-e", "--epsilon", dest="epsilon", type=float, help="epsilon value", default=0.1
    )
    parser.add_argument(
        "-s", "--seed", dest="seed_idx", type=int, help="Variable to pass seed index", default=None
    )
    parser.add_argument(
        "-r",
        "--reshaping",
        dest="reshaping",
        nargs="+",
        type=int,
        help="Determines the reshaping for the tensor stream",
        default=[],
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", type=int, help="Batch size", default=1
    )
    parser.add_argument(
        "-n",
        "--numpy",
        dest="numpy",
        action="store_true",
        help="Read the images from numpy files",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Use wandb for logging",
        default=False,
    )
    return parser.parse_args()


def compress_BasaltMineRL_TT():
    args = get_args()
    filter = [True, True, True, True, True, True, True, True, True, True, True, True]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
    else:
        seed_idx = args.seed_idx

    if args.wandb:
        run_config = wandb.config = {
            "epsilon": args.epsilon,
            "batch_size": args.batch_size,
            "reshaping": args.reshaping,
            "numpy": args.numpy,
            "filter": filter,
            "seed_idx": seed_idx,
        }
        wandb.init(
            # entity="dorukaks",
            project="HierarchicalTucker_experiments",
            name="BigEarthNet_TT_eps_"
            + "".join(f"{args.epsilon:0.2f}_".split("."))
            + f"batchsize_{args.batch_size:04d}_"
            + "shape_"
            + "_".join(map(str, args.reshaping))
            + "_date_"
            + timestamp,
            config=run_config,
            tags=["BigEarthNet", "TT-ICE", MACHINE_ALIAS],
        )
    print("Seed index is: ", seed_idx)
    random.seed(seed_idx)  # Fix the seed to a random number
    np.random.seed(seed_idx)  # Fix the seed to a random number
    print(args)
    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for BigEarthNet")
        args.reshaping = [12, 10, 12, 10, 12]
    args.reshaping[-1] = sum(filter)
    print("Reshaping used: ", args.reshaping)

    if args.data_location is None:
        if args.numpy:
            if os.path.exists(
                reduce(
                    os.path.join,
                    [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0_numpy"],
                )
            ):
                print("Data location is not provided, using default location")
                data_loc = reduce(
                    os.path.join,
                    [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0_numpy"],
                )
            else:
                raise IsADirectoryError("Please provide the data location")
        else:
            if os.path.exists(
                reduce(
                    os.path.join,
                    [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0"],
                )
            ):
                print("Data location is not provided, using default location")
                data_loc = reduce(
                    os.path.join,
                    [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0"],
                )
            else:
                raise IsADirectoryError("Please provide the data location")
    else:
        data_loc = args.data_location
        if args.numpy:
            assert glob.glob(
                args.data_location + "*.npy"
            ), "There are no numpy files in the provided directory."

    directories = {
        "data": data_loc + PATH_SEP,
        "cores": reduce(os.path.join, [PATH_SEP] + CWD.split(os.path.sep) + ["savedCores"])
        + PATH_SEP,
        "metric": reduce(
            os.path.join, [PATH_SEP] + CWD.split(os.path.sep) + ["experiments", "BigEarthNet"]
        )
        + PATH_SEP,
    }

    if not os.path.exists(directories["cores"]):
        os.makedirs(directories["cores"])
    if not os.path.exists(directories["metric"]):
        os.makedirs(directories["metric"])

    image_folders = glob.glob(directories["data"] + "*")
    num_images = len(image_folders)
    print("Number of images: ", num_images)
    random.shuffle(image_folders)

    train_images = image_folders[: int(num_images * TRAIN_RATIO)]
    image_folders = image_folders[int(num_images * TRAIN_RATIO) :]
    val_images = image_folders[: int(num_images * VAL_RATIO)]
    image_folders = image_folders[int(num_images * VAL_RATIO) :]
    test_images = image_folders[: int(num_images * TEST_RATIO)]
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
            train_data = np.load(image_dir)[..., filter].reshape(args.reshaping)[..., None]
            train_batch.append(train_data)
        # toc = time.time() - tic
        train_batch = np.concatenate(train_batch, axis=-1)
        batch_norm = nla.norm(train_batch)
        # batch_along = len(train_batch.shape) - 1

        dataset = dmt.ttObject(
            train_batch,
            epsilon=args.epsilon,
            keepData=False,
            samplesAlongLastDimension=True,
            method="ttsvd",
        )

        tic = time.time()
        dataset.ttDecomp(dtype=np.float64)
        batch_time = time.time() - tic
        total_time += batch_time

        rec = dataset.reconstruct(dataset.ttCores[-1][:, -args.batch_size :, :]).squeeze()
        error_before_update = 0
        error_after_update = nla.norm(rec - train_batch) / batch_norm
        # train_images = train_images[:100*args.batch_size]
        val_errors = []
        for image in val_images:
            val_data = np.load(image)[..., filter].reshape(args.reshaping)[..., None]
            rec = dataset.reconstruct(
                dataset.projectTensor(
                    val_data,
                )
            )
            approximation_error = nla.norm(rec - val_data) / nla.norm(val_data)
            val_errors.append(approximation_error)
            # print(approximation_error)
        test_errors = []
        for image in test_images:
            test_data = np.load(image)[..., filter].reshape(args.reshaping)[..., None]
            rec = dataset.reconstruct(
                dataset.projectTensor(
                    test_data,
                )
            ).squeeze()
            approximation_error = nla.norm(rec - test_data.squeeze()) / nla.norm(test_data)
            test_errors.append(approximation_error)
        ranks = dataset.ttRanks[1:-1]
        # for core in dataset.transfer_nodes:
        #     ranks.append(core.shape[-1])
        # for leaf in dataset.leaves:
        #     ranks.append(leaf.shape[-1])
        print(
            f"{batch_index:04d} {dataset.ttCores[-1].shape[1]:06d} {round(batch_time, 5):08.5f} \
                {round(total_time, 5):09.5f} {batch_norm:14.5f} \
                    {round(error_before_update, 5):0.5f} \
                    {round(error_after_update, 5):0.5f} \
                        {round(dataset.compressionRatio, 5):09.5f} \
                        {round(np.mean(val_errors),5):0.5f} \
                            {round(np.mean(test_errors),5):0.5f} \
                            {' '.join(map(lambda x: f'{x:04d}', ranks))}"
        )
        logging_dict = {
            "compression_ratio": dataset.compressionRatio,
            "error_before_update": error_before_update,
            "error_after_update": error_after_update,
            "image_count": dataset.ttCores[-1].shape[1],
            "batch_time": batch_time,
            "total_time": total_time,
            "val_error": np.mean(val_errors),
            "test_error": np.mean(test_errors),
            "batch_norm": batch_norm,
        }
        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank

        if args.wandb:
            wandb.log(logging_dict)

        previous_ranks = dataset.ttRanks.copy()
        # train_images = train_images[:124]

        while train_images:
            batch_index += 1
            train_batch = []
            for _ in range(args.batch_size):
                try:
                    image_dir = train_images.pop(0)
                except Exception:
                    break
                train_data = np.load(image_dir)[..., filter].reshape(args.reshaping)[..., None]
                train_batch.append(train_data)
            train_batch = np.concatenate(train_batch, axis=-1)
            batch_norm = nla.norm(train_batch)
            projection = dataset.reconstruct(
                dataset.projectTensor(
                    train_batch,
                )
            ).squeeze()
            error_before_update = nla.norm(projection - train_batch) / batch_norm
            tic = time.time()
            dataset.ttICE(
                train_batch,
                epsilon=args.epsilon,
            )
            batch_time = time.time() - tic
            update_flag = dataset.ttRanks != previous_ranks
            rec = dataset.reconstruct(
                dataset.ttCores[-1][:, -train_batch.shape[-1] :, :]
                # dataset.root.core[...,-args.batch_size:]
            ).squeeze()
            error_after_update = nla.norm(rec - train_batch) / batch_norm
            total_time += batch_time
            if update_flag:
                val_errors = []
                for image in val_images:
                    val_data = np.load(image)[..., filter].reshape(args.reshaping)[..., None]
                    rec = dataset.reconstruct(
                        dataset.projectTensor(
                            val_data,
                        )
                    ).squeeze()
                    approximation_error = nla.norm(rec - val_data.squeeze()) / nla.norm(val_data)
                    val_errors.append(approximation_error)
                test_errors = []
                for image in test_images:
                    test_data = np.load(image)[..., filter].reshape(args.reshaping)[..., None]
                    rec = dataset.reconstruct(
                        dataset.projectTensor(
                            test_data,
                        )
                    ).squeeze()
                    approximation_error = nla.norm(rec - test_data.squeeze()) / nla.norm(test_data)
                    test_errors.append(approximation_error)
                ranks = dataset.ttRanks[1:-1]
                # for core in dataset.transfer_nodes:
                #     ranks.append(core.shape[-1])
                # for leaf in dataset.leaves:
                #     ranks.append(leaf.shape[-1])

            logging_dict = {
                "compression_ratio": dataset.compressionRatio,
                "error_before_update": error_before_update,
                "error_after_update": error_after_update,
                "image_count": dataset.ttCores[-1].shape[1],
                "batch_time": batch_time,
                "total_time": total_time,
                "val_error": np.mean(val_errors),
                "test_error": np.mean(test_errors),
                "batch_norm": batch_norm,
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            if args.wandb:
                wandb.log(logging_dict)

            print(
                f"{batch_index:04d} {dataset.ttCores[-1].shape[1]:06d} \
                    {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} \
                        {batch_norm:14.5f} {round(error_before_update, 5):0.5f} \
                        {round(error_after_update, 5):0.5f} \
                            {round(dataset.compressionRatio, 5):09.5f} \
                            {round(np.mean(val_errors),5):0.5f} \
                                {round(np.mean(test_errors),5):0.5f} \
                                {' '.join(map(lambda x: f'{x:04d}', ranks))}"
            )
            # {' '.join(map(lambda x: f'{x:03d}', ranks))}
            previous_ranks = dataset.ttRanks.copy()

        numel = 0
        for core in dataset.ttCores:
            numel += np.prod(core.shape)
            print(core.shape)
        # numel = np.prod(dataset.root.core.shape)
        # print(dataset.root.shape)
        # for core in dataset.transfer_nodes:
        #     print(core.shape)
        #     numel += np.prod(core.shape)
        # for leaf in dataset.leaves:
        #     print(leaf.shape)
        #     numel += np.prod(leaf.shape)
        print(numel)
        print(np.prod(dataset.originalShape))
        print(dataset.ttCores[-1].shape[1])
        print(dataset.originalShape)

    else:
        raise NotImplementedError("Only numpy files are supported for now")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    compress_BasaltMineRL_TT()
