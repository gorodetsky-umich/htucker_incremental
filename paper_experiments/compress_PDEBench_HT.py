#!.env/bin/python -u
import argparse
import datetime
import glob
import os
import random
import time
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
import numpy.linalg as nla
from experiment_utils import get_args_pdebench, normalize

import htucker as ht
import wandb

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
ORD = "F"
STATES = [
    # 'Vx', 'Vy', 'Vz', 'density', 'pressure'
    "Vx",
    "Vy",
    "Vz",
]
MACHINE_ALIAS = "LH"

__all__ = [
    "get_args",
    "initialize_wandb",
    "read_snapshot",
]


def read_simulations(simulation_idx, states, timesteps=None, path_to_data="./", file_name=""):
    sim = []
    # print(path_to_data,file_name, simulation_idx, timesteps)
    with h5py.File(os.path.join(path_to_data, file_name), "r") as f:
        for state in states:
            # print(state, f[state].shape)
            if timesteps is None:
                # Read all timesteps
                sim.append(f[state][simulation_idx, :, ...].transpose(1, 2, 3, 0)[..., None])
            else:
                # Read only the specified timestep(s)
                sim.append(f[state][simulation_idx, timesteps, ...][..., None])
    return np.concatenate(sim, axis=-1)[..., None]


def get_args():
    parser = argparse.ArgumentParser(
        description="This script reads the PDEBench simulation snapshots and \
            compresses them using the HT format."
    )
    parser.add_argument(
        "-s", "--seed", dest="seed_idx", type=int, help="Variable to pass seed index", default=None
    )
    parser.add_argument(
        "-e", "--epsilon", dest="epsilon", type=float, help="epsilon value", default=0.1
    )
    parser.add_argument(
        "-d", "--data_location", dest="data_location", help="path to data", default=None
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
        "--states",
        dest="states",
        nargs="+",
        type=int,
        help="Determines the states that will be compressed",
        default=list(range(len(STATES))),
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", type=int, help="Batch size", default=1
    )
    parser.add_argument(
        "-w",
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Use wandb for logging",
        default=False,
    )
    parser.add_argument(
        "-t", "--type", dest="type", type=str, help="Type of simulation data", default="Rand"
    )
    parser.add_argument(
        "-c",
        "--combine",
        dest="combine",
        action="store_true",
        help="Combine timesteps of the simulation",
        default=False,
    )
    parser.add_argument(
        "-n",
        "--numpy",
        dest="numpy",
        action="store_true",
        help="Use extracted numpy files to read data",
        default=False,
    )
    parser.add_argument(
        "-M", "--mach_number", dest="M", help="Mach number for the simulations", default=None
    )
    return parser.parse_args()


def initialize_wandb(args, timestamp, method, tags):
    run_config = wandb.config = {
        "seed_idx": args.seed_idx,
        "epsilon": args.epsilon,
        "normalization": args.normalization,
        "simulation_type": args.type,
        "mach_number": args.M,
        "states": args.states,
        "filter": filter,
        "reshaping": args.reshaping,
        "transpose": args.transpose,
        "batch_size": args.batch_size,
        "numpy": args.numpy,
    }
    wandb.init(
        project="HierarchicalTucker_experiments",
        name=f"PDEBench_{method}_eps_"
        + "".join(f"{args.epsilon:0.2f}_".split("."))
        + f"{args.normalization}_batchsize_{args.batch_size:04d}_"
        + "shape_"
        + "_".join(map(str, args.reshaping))
        + "_states_"
        + "_".join(map(str, args.states))
        + "_date_"
        + timestamp,
        config=run_config,
        tags=tags,
    )


def read_snapshot(timestep, simulation, states, data_loc):
    return np.load(data_loc + f"sim{simulation:03d}_ts{timestep:02d}.npy")[:, :, :, states][
        ..., None
    ]


def numpy_combine(args):
    np.set_printoptions(linewidth=100)

    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    num_states = len(args.states)
    data_loc = args.data_location + PATH_SEP + f"{args.type}" + PATH_SEP + f"{args.M}" + PATH_SEP
    print(data_loc)
    num_simulations = len(glob.glob(data_loc + "sim*_ts00.npy"))
    num_timesteps = len(glob.glob(data_loc + "sim000_ts*.npy"))
    grid_shape = list(np.load(data_loc + "sim000_ts00.npy").shape[:-1])
    print(num_simulations, num_timesteps, grid_shape)
    assert np.prod(grid_shape) == np.prod(
        args.reshaping
    ), "Reshaping does not match the shape of the data"
    if args.transpose is not None:
        assert len(args.transpose) == (
            len(args.reshaping) + 3
        ), f"Transpose dimensions {args.transpose} are not in aggreement with the \
            shape of data {args.reshaping+[num_states,num_timesteps,args.batch_size]}"

    if args.wandb:
        initialize_wandb(
            args,
            timestamp,
            method="HT",
            tags=["PDEBench", "HTucker", MACHINE_ALIAS, args.M, args.type],
        )

    train_simulations = random.sample(range(0, num_simulations), int(num_simulations * TRAIN_RATIO))
    val_simulations = random.sample(
        list(set(range(0, num_simulations)) - set(train_simulations)),
        int(num_simulations * VAL_RATIO),
    )
    if TEST_RATIO == 0:
        test_simulations = []
    else:
        test_simulations = list(
            set(range(0, num_simulations)) - set(train_simulations) - set(val_simulations)
        )
    total_time = 0
    batch_index = 0
    print(
        f"Number of training simulations: {len(train_simulations)}\nNumber of validation \
            simulations:{len(val_simulations)}\n\
                Number of testing simulations:{len(test_simulations)}"
    )
    tic = time.time()
    train_batch = []
    # for sim_idx, simulation in enumerate(train_simulations[:args.batch_size]):
    normalizing_constants_batch = []
    for simulation in train_simulations[: args.batch_size]:
        normalizing_constants_sim = []
        with Pool() as p:
            sim = p.map(
                partial(
                    read_snapshot, data_loc=data_loc, simulation=simulation, states=args.states
                ),
                list(range(num_timesteps)),
            )
        sim = np.concatenate(sim, axis=-1)
        for state_idx in range(num_states):
            sim[..., state_idx, :], normalizing_constant1, normalizing_constant2 = normalize(
                sim[..., state_idx, :], method=args.normalization
            )
            normalizing_constants_sim.append(
                np.array([normalizing_constant1, normalizing_constant2])[..., None]
            )
        # train_batch.append(np.concatenate(sim,axis=-1).transpose(0,1,2,4,3)[...,None])
        train_batch.append(sim[..., None])
        # if args.transpose:
        #     train_batch = np.transpose(train_batch,args.transpose)
        normalizing_constants_batch.append(np.concatenate(normalizing_constants_sim, axis=-1).T)
    train_batch = np.concatenate(train_batch, axis=-1)
    print(train_batch.shape)

    train_batch = train_batch.reshape(
        args.reshaping + [num_states, num_timesteps, args.batch_size], order=ORD
    )
    if args.transpose:
        train_batch = np.transpose(train_batch, args.transpose)
    print(train_batch.shape)
    batch_norm = nla.norm(train_batch)

    batch_along = len(train_batch.shape) - 1
    dataset = ht.HTucker()
    dataset.initialize(train_batch, batch=True, batch_dimension=batch_along)
    dataset.normalizing_constants = normalizing_constants_batch.copy()
    dimension_tree = ht.createDimensionTree(
        dataset.original_shape[:batch_along] + dataset.original_shape[batch_along + 1 :],
        numSplits=2,
        minSplitSize=1,
    )
    dimension_tree.get_items_from_level()
    dataset.rtol = args.epsilon

    tic = time.time()
    dataset.compress_leaf2root_batch(
        train_batch,
        dimension_tree=dimension_tree,
        batch_dimension=batch_along,
    )
    batch_time = time.time() - tic
    total_time += batch_time

    rec = dataset.reconstruct(dataset.root.core[..., -args.batch_size :])

    error_before_update = 0
    error_after_update = nla.norm(rec - train_batch) / batch_norm

    val_errors = []
    test_errors = []

    for simulation in val_simulations:
        with Pool() as p:
            val_data = p.map(
                partial(
                    read_snapshot, data_loc=data_loc, simulation=simulation, states=args.states
                ),
                list(range(num_timesteps)),
            )
        # val_data = np.concatenate(val_data,axis=-1).transpose(0,1,2,4,3)[...,None]
        val_data = np.concatenate(val_data, axis=-1)
        for state_idx in range(num_states):
            val_data[..., state_idx, :], _, _ = normalize(
                val_data[..., state_idx, :], method=args.normalization
            )
        val_data = val_data[..., None]
        # val_data = val_data.reshape(args.reshaping+[num_timesteps,num_states,1], order=ORD)
        val_data = val_data.reshape(args.reshaping + [num_states, num_timesteps, 1], order=ORD)
        if args.transpose:
            val_data = np.transpose(val_data, args.transpose)
        rec = dataset.reconstruct(
            dataset.project(val_data, batch=True, batch_dimension=batch_along)
        )
        approximation_error = nla.norm(rec - val_data) / nla.norm(val_data)
        val_errors.append(approximation_error)

    for simulation in test_simulations:
        with Pool() as p:
            test_data = p.map(
                partial(
                    read_snapshot, data_loc=data_loc, simulation=simulation, states=args.states
                ),
                list(range(num_timesteps)),
            )
        test_data = np.concatenate(test_data, axis=-1)
        for state_idx in range(num_states):
            test_data[..., state_idx, :], _, _ = normalize(
                test_data[..., state_idx, :], method=args.normalization
            )
        # test_data = np.concatenate(test_data,axis=-1).transpose(0,1,2,4,3)[...,None]
        test_data = test_data[..., None]
        # test_data = test_data.reshape(args.reshaping+[num_timesteps,num_states,1], order=ORD)
        test_data = test_data.reshape(args.reshaping + [num_states, num_timesteps, 1], order=ORD)
        if args.transpose:
            test_data = np.transpose(test_data, args.transpose)
        rec = dataset.reconstruct(
            dataset.project(test_data, batch=True, batch_dimension=batch_along)
        )
        approximation_error = nla.norm(rec - test_data) / nla.norm(test_data)
        test_errors.append(approximation_error)

    ranks = []
    for core in dataset.transfer_nodes:
        ranks.append(core.shape[-1])
    for leaf in dataset.leaves:
        ranks.append(leaf.shape[-1])
    print(
        f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
    )
    if args.wandb:
        logging_dict = {
            "compression_ratio": dataset.compression_ratio,
            "error_before_update": error_before_update,
            "error_after_update": error_after_update,
            "image_count": dataset.root.core.shape[-1],
            "batch_time": batch_time,
            "total_time": total_time,
            "val_error": np.mean(val_errors),
            "test_error": np.mean(test_errors),
            "batch_norm": batch_norm,
        }
        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank
        wandb.log(logging_dict)
    train_simulations = train_simulations[args.batch_size :]
    # ADDED SAVING TO EXTRACT TURBULENCE SPECTRUM #
    # epsilon_str = "".join(f"{args.epsilon:0.2f}".split("."))
    dataset.seed = args.seed_idx
    # dataset.save(f'PDEBench3_{args.normalization}_e{epsilon_str}',directory='/nfs/turbo/coe-goroda/doruk_temp/')

    while train_simulations[:]:
        batch_index += 1

        normalizing_constants_batch = []
        train_batch = []
        # for sim_idx, simulation in enumerate(train_simulations[:args.batch_size]):
        for simulation in train_simulations[: args.batch_size]:
            normalizing_constants_sim = []
            with Pool() as p:
                sim = p.map(
                    partial(
                        read_snapshot, data_loc=data_loc, simulation=simulation, states=args.states
                    ),
                    list(range(num_timesteps)),
                )
            sim = np.concatenate(sim, axis=-1)
            for state_idx in range(num_states):
                sim[..., state_idx, :], normalizing_constant1, normalizing_constant2 = normalize(
                    sim[..., state_idx, :], method=args.normalization
                )
                normalizing_constants_sim.append(
                    np.array([normalizing_constant1, normalizing_constant2])[..., None]
                )
            # if args.transpose:
            #     train_batch = np.transpose(train_batch,args.transpose)
            train_batch.append(sim[..., None])
            normalizing_constants_batch.append(np.concatenate(normalizing_constants_sim, axis=-1).T)
        train_batch = np.concatenate(train_batch, axis=-1)
        train_batch = train_batch.reshape(
            args.reshaping + [num_states, num_timesteps, args.batch_size], order=ORD
        )
        if args.transpose:
            train_batch = np.transpose(train_batch, args.transpose)
        dataset.normalizing_constants.append(normalizing_constants_batch.copy())
        batch_norm = nla.norm(train_batch)

        projection = dataset.reconstruct(
            dataset.project(train_batch, batch=True, batch_dimension=batch_along)
        )
        error_before_update = nla.norm(projection - train_batch) / batch_norm
        tic = time.time()
        update_flag = dataset.incremental_update_batch(
            train_batch,
            batch_dimension=batch_along,
            append=True,
        )
        batch_time = time.time() - tic
        rec = dataset.reconstruct(
            dataset.root.core[..., -train_batch.shape[-1] :]
            # dataset.root.core[...,-args.batch_size:]
        )
        error_after_update = nla.norm(rec - train_batch) / batch_norm
        total_time += batch_time

        if update_flag:
            val_errors = []
            test_errors = []
            for simulation in val_simulations:
                with Pool() as p:
                    val_data = p.map(
                        partial(
                            read_snapshot,
                            data_loc=data_loc,
                            simulation=simulation,
                            states=args.states,
                        ),
                        list(range(num_timesteps)),
                    )
                # val_data = np.concatenate(val_data,axis=-1).transpose(0,1,2,4,3)[...,None]
                val_data = np.concatenate(val_data, axis=-1)
                for state_idx in range(num_states):
                    val_data[..., state_idx, :], _, _ = normalize(
                        val_data[..., state_idx, :], method=args.normalization
                    )
                val_data = val_data[..., None]
                val_data = val_data.reshape(
                    args.reshaping + [num_states, num_timesteps, 1], order=ORD
                )
                if args.transpose:
                    val_data = np.transpose(val_data, args.transpose)
                rec = dataset.reconstruct(
                    dataset.project(val_data, batch=True, batch_dimension=batch_along)
                )
                approximation_error = nla.norm(rec - val_data) / nla.norm(val_data)
                val_errors.append(approximation_error)

            for simulation in test_simulations:
                with Pool() as p:
                    test_data = p.map(
                        partial(
                            read_snapshot,
                            data_loc=data_loc,
                            simulation=simulation,
                            states=args.states,
                        ),
                        list(range(num_timesteps)),
                    )
                test_data = np.concatenate(test_data, axis=-1)
                for state_idx in range(num_states):
                    test_data[..., state_idx, :], _, _ = normalize(
                        test_data[..., state_idx, :], method=args.normalization
                    )
                test_data = test_data[..., None]
                test_data = test_data.reshape(
                    args.reshaping + [num_states, num_timesteps, 1], order=ORD
                )
                if args.transpose:
                    test_data = np.transpose(test_data, args.transpose)
                rec = dataset.reconstruct(
                    dataset.project(test_data, batch=True, batch_dimension=batch_along)
                )
                approximation_error = nla.norm(rec - test_data) / nla.norm(test_data)
                test_errors.append(approximation_error)

            ranks = []
            for core in dataset.transfer_nodes:
                ranks.append(core.shape[-1])
            for leaf in dataset.leaves:
                ranks.append(leaf.shape[-1])
        print(
            f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )

        # ADDED SAVING TO EXTRACT TURBULENCE SPECTRUM #
        # epsilon_str = "".join(f"{args.epsilon:0.2f}".split("."))
        dataset.seed = args.seed_idx

        if args.wandb:
            logging_dict = {
                "compression_ratio": dataset.compression_ratio,
                "error_before_update": error_before_update,
                "error_after_update": error_after_update,
                "image_count": dataset.root.core.shape[-1],
                "batch_time": batch_time,
                "total_time": total_time,
                "val_error": np.mean(val_errors),
                "test_error": np.mean(test_errors),
                "batch_norm": batch_norm,
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            wandb.log(logging_dict)
        train_simulations = train_simulations[args.batch_size :]
    numel = np.prod(dataset.root.core.shape)

    print(dataset.root.shape)
    for core in dataset.transfer_nodes:
        print(core.shape)
        numel += np.prod(core.shape)
    for leaf in dataset.leaves:
        print(leaf.shape)
        numel += np.prod(leaf.shape)
    print(numel)
    print(np.prod(train_batch.shape[:-1]), dataset.root.shape[-1])
    print(dataset.batch_count)
    print(dataset.original_shape)
    return


def numpy_nocombine(args):
    # args = get_args()
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # num_states = len(STATES)
    # data_loc = args.data_location + PATH_SEP + f"{args.type}" + PATH_SEP + f"{args.M}" + PATH_SEP
    # num_simulations = len(glob.glob(data_loc + "sim*_ts00.npy"))
    # num_timesteps = len(glob.glob(data_loc + "sim000_ts*.npy"))
    raise NotImplementedError("This method is not yet implemented")

    return


def hdf5_combine(args):
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # num_states = len(STATES)
    # data_loc = args.data_location + PATH_SEP + f"{args.type}" + PATH_SEP + f"{args.M}" + PATH_SEP
    # num_simulations = len(glob.glob(data_loc + "sim*_ts00.npy"))
    # num_timesteps = len(glob.glob(data_loc + "sim000_ts*.npy"))
    raise NotImplementedError("This method is not yet implemented")


def hdf5_nocombine(args):
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # num_states = len(STATES)
    # data_loc = args.data_location + PATH_SEP + f"{args.type}" + PATH_SEP + f"{args.M}" + PATH_SEP
    # num_simulations = len(glob.glob(data_loc + "sim*_ts00.npy"))
    # num_timesteps = len(glob.glob(data_loc + "sim000_ts*.npy"))
    raise NotImplementedError("This method is not yet implemented")


if __name__ == "__main__":

    args = get_args_pdebench()

    if args.type in ["rand", "random", "randomized", "Rand", "RAND", "Random"]:
        args.type = "Rand"
    elif args.type in [
        "turb",
        "turbulence",
        "turbulent",
        "Turb",
        "TURB",
        "Turbulence",
        "Turbulent",
    ]:
        args.type = "Turb"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if (args.type == "Rand") and (args.M is None):
        print("No M is given. Using default M=1.0")
        args.M = "1.0"
    elif (args.type == "Turb") and (args.M is None):
        args.M = "1.0"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass

    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for PDEBench")
        if args.type == "Rand":
            # args.reshaping = [8,4,4,8,4,4,8,4,4]
            args.reshaping = [8, 4, 4, 8, 4, 4, 8, 4, 4]
        elif args.type == "Turb":
            # args.reshaping = [8,8,8,8,8,8]
            args.reshaping = [8, 8, 8, 8, 8, 8]
    # args.reshaping.extend([sum(args.states)])
    # print("Reshaping used: ", args.reshaping)

    print(args)
    overall_tic = time.time()
    if args.numpy and args.combine:
        print("numpy+combine")
        numpy_combine(args)
    elif args.numpy:
        print("numpy+not combine")
        numpy_nocombine(args)
    elif args.combine:
        print("combine")
        hdf5_combine(args)
    else:
        print("nothing")
        hdf5_nocombine(args)

    overall_time = time.time() - overall_tic
    print(f"Time for entire process (s): {round(overall_time,3)}")
