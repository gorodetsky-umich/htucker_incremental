#!.env/bin/python
import datetime
import glob
import os
import sys
import time

import numpy as np

import wandb

sys.path.insert(0, "/home/doruk/TT-ICE/")
import DaMAT as dmt
import numpy.linalg as nla
from experiment_utils import get_args_catgel, initialize_wandb_catgel, normalize

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"
MACHINE_ALIAS = "LH"
HEUR_2_USE = ["skip", "occupancy"]
OCCUPANCY = 1


def compress_catgel_full_sim_TT(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if args.wandb:
        initialize_wandb_catgel(
            args, timestamp, tags=["CatGel", "TT-ICE", MACHINE_ALIAS], method="TT"
        )

    data_loc = args.data_location
    training_simulations = glob.glob(data_loc + f"/3rd/{args.type}/*.npy")
    training_simulations.sort()
    test_simulations = glob.glob(data_loc + f"/5th/{args.type}/*.npy")

    normalizing_constants = []
    training_snapshot = np.load(training_simulations[0])
    for state_idx in range(training_snapshot.shape[-1]):
        training_snapshot[..., state_idx], normalizing_constant1, normalizing_constant2 = normalize(
            training_snapshot[..., state_idx], method=args.normalization
        )
        normalizing_constants.append(
            np.array([normalizing_constant1, normalizing_constant2])[..., None]
        )

    total_time = 0
    batch_index = 0

    training_snapshot = training_snapshot.transpose(0, 1, 2, 4, 3)[..., None]
    batch_norm = nla.norm(training_snapshot)

    # batch_along = (
    #     len(training_snapshot.shape) - 1
    # )  # No -1 needed here since the batch dimension is created afterwards

    dataset = dmt.ttObject(
        training_snapshot,
        epsilon=args.epsilon,
        keepData=False,
        samplesAlongLastDimension=True,
        method="ttsvd",
    )

    tic = time.time()
    dataset.ttDecomp(dtype=np.float64)
    batch_time = time.time() - tic
    total_time += batch_time
    previous_ranks = dataset.ttRanks.copy()

    rec = dataset.reconstruct(dataset.ttCores[-1][:, -training_snapshot.shape[-1] :, :]).squeeze(0)[
        ..., 0
    ]

    error_before_update = 0
    error_after_update = nla.norm(rec - training_snapshot) / batch_norm

    test_errors = []
    for simulation in test_simulations[:]:
        test_snapshot = np.load(simulation)
        for state_idx in range(test_snapshot.shape[-1]):
            test_snapshot[..., state_idx], _, _ = normalize(
                test_snapshot[..., state_idx], method=args.normalization
            )
        test_snapshot = test_snapshot.transpose(0, 1, 2, 4, 3)[..., None]
        rec = dataset.reconstruct(
            dataset.projectTensor(
                test_snapshot,
            )
        ).squeeze(0)
        approximation_error = nla.norm(rec - test_snapshot) / nla.norm(test_snapshot)
        test_errors.append(approximation_error)

    ranks = dataset.ttRanks[1:-1].copy()
    print(
        f"{batch_index:04d} {dataset.ttCores[-1].shape[1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
    )

    if args.wandb:
        logging_dict = {
            "compression_ratio": dataset.compressionRatio,
            "error_before_update": error_before_update,
            "error_after_update": error_after_update,
            "simulation_count": dataset.ttCores[-1].shape[1],
            "batch_time": batch_time,
            "total_time": total_time,
            "test_error": np.mean(test_errors),
            "batch_norm": batch_norm,
        }
        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank
        wandb.log(logging_dict)

    training_simulations = training_simulations[1:]

    while training_simulations[:]:
        training_simulation = training_simulations.pop(0)
        batch_index += 1
        normalizing_constants = []
        training_snapshot = np.load(training_simulation)
        for state_idx in range(training_snapshot.shape[-1]):
            training_snapshot[..., state_idx], normalizing_constant1, normalizing_constant2 = (
                normalize(training_snapshot[..., state_idx], method=args.normalization)
            )
            normalizing_constants.append(
                np.array([normalizing_constant1, normalizing_constant2])[..., None]
            )
        training_snapshot = training_snapshot.transpose(0, 1, 2, 4, 3)[..., None]
        batch_norm = nla.norm(training_snapshot)
        # print(training_snapshot.shape)
        projection = dataset.reconstruct(
            dataset.projectTensor(
                training_snapshot,
            )
        ).squeeze(0)
        error_before_update = nla.norm(projection - training_snapshot) / batch_norm
        tic = time.time()
        dataset.ttICEstar(
            training_snapshot,
            heuristicsToUse=HEUR_2_USE,
            epsilon=args.epsilon,
            occupancyThreshold=OCCUPANCY,
        )
        batch_time = time.time() - tic
        rec = dataset.reconstruct(
            dataset.ttCores[-1][..., -training_snapshot.shape[-1] :, 0]
        ).squeeze(0)
        error_after_update = nla.norm(rec - training_snapshot) / batch_norm
        total_time += batch_time

        update_flag = dataset.ttRanks != previous_ranks
        # print(update_flag)
        if update_flag:
            for simulation in test_simulations[:]:
                test_snapshot = np.load(simulation)
                for state_idx in range(test_snapshot.shape[-1]):
                    test_snapshot[..., state_idx], _, _ = normalize(
                        test_snapshot[..., state_idx], method=args.normalization
                    )
                test_snapshot = test_snapshot.transpose(0, 1, 2, 4, 3)[..., None]
                rec = dataset.reconstruct(
                    dataset.projectTensor(
                        test_snapshot,
                    )
                ).squeeze(0)
                approximation_error = nla.norm(rec - test_snapshot) / nla.norm(test_snapshot)
                test_errors.append(approximation_error)

            ranks = dataset.ttRanks[1:-1].copy()

        previous_ranks = dataset.ttRanks.copy()
        print(
            f"{batch_index:04d} {dataset.ttCores[-1].shape[-1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )
        if args.wandb:
            logging_dict = {
                "compression_ratio": dataset.compressionRatio,
                "error_before_update": error_before_update,
                "error_after_update": error_after_update,
                "simulation_count": dataset.ttCores[-1].shape[1],
                "batch_time": batch_time,
                "total_time": total_time,
                "test_error": np.mean(test_errors),
                "batch_norm": batch_norm,
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            wandb.log(logging_dict)


if __name__ == "__main__":
    args = get_args_catgel()

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass

    print(args)
    overall_tic = time.time()
    compress_catgel_full_sim_TT(args)
    overall_time = time.time() - overall_tic
    print(f"Time for entire process (s): {round(overall_time,3)}")
