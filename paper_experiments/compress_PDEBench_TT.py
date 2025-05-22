#!.env/bin/python -u
import datetime
import glob
import os
import random
import sys
import time

import wandb

sys.path.insert(0, "/home/doruk/TT-ICE/")
from functools import partial, reduce
from multiprocessing import Pool

import DaMAT as dmt
import numpy as np
import numpy.linalg as nla
from compress_PDEBench_HT import get_args, initialize_wandb, read_snapshot
from experiment_utils import get_args_pdebench, normalize

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.8
VAL_RATIO = 0  # 0.1
TEST_RATIO = 0  # 0.1
ORD = "F"
HEUR_2_USE = ["skip", "occupancy"]
STATES = [
    # 'Vx', 'Vy', 'Vz', 'density', 'pressure'
    "Vx",
    "Vy",
    "Vz",
]
OCCUPANCY = 1
MACHINE_ALIAS = "LH"


def main():
    args = get_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

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
    print(args)

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
    else:
        seed_idx = args.seed_idx

    # states = ["Vx", "Vy", "Vz", "density", "pressure"]
    # num_states = len(states)

    # if args.type == "Rand":
    #     M = "0.1"
    # elif args.type == "Turb":
    #     M = "1.0"
    # else:
    #     raise ValueError("Type of simulation is not recognized.")

    # file_name = f"3D_CFD_{args.type}_M{M}_Eta1e-08_Zeta1e-08_periodic_train.hdf5"
    random.seed(seed_idx)  # Fix the seed to a random number
    np.random.seed(seed_idx)  # Fix the seed to a random number

    print("Seed index is: ", seed_idx)
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
            name="PDEBench_TT_eps_"
            + "".join(f"{args.epsilon:0.2f}_".split("."))
            + f"batchsize_{args.batch_size:04d}_"
            + "shape_"
            + "_".join(map(str, args.reshaping))
            + "_date_"
            + timestamp,
            config=run_config,
            tags=["PDEBench", "TT-ICE", "MBP23"],
        )

    # print(args.reshaping)
    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for PDEBench")
        if args.type == "Rand":
            # args.reshaping = [8,4,4,8,4,4,8,4,4]
            args.reshaping = [8, 4, 4, 8, 4, 4, 8, 4, 4]
        elif args.type == "Turb":
            # args.reshaping = [8,8,8,8,8,8]
            args.reshaping = [8, 8, 8, 8, 8, 8]
    # args.reshaping[-1] = sum(filter)
    print("Reshaping used: ", args.reshaping)

    if args.data_location is None:
        # if args.numpy:
        #     if os.path.exists(reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data"])):
        #         print("Data location is not provided, using default location")
        #         data_loc = reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data"])
        #     else:
        #         raise IsADirectoryError("Please provide the data location")
        # else:
        if os.path.exists(reduce(os.path.join, [PATH_SEP] + HOME.split(os.path.sep) + ["data"])):
            print("Data location is not provided, using default location")
            data_loc = reduce(os.path.join, [PATH_SEP] + HOME.split(os.path.sep) + ["data"])
        else:
            raise IsADirectoryError("Please provide the data location")
    else:
        data_loc = args.data_location
        # if args.numpy:
        #     assert glob.glob(args.data_location+"*.npy"), "There are no numpy \
        #       files in the provided directory."

    directories = {
        "data": data_loc + PATH_SEP,
        "cores": reduce(os.path.join, [PATH_SEP] + CWD.split(os.path.sep) + ["savedCores"])
        + PATH_SEP,
        "metric": reduce(
            os.path.join, [PATH_SEP] + CWD.split(os.path.sep) + ["experiments", "BigEarthNet"]
        )
        + PATH_SEP,
    }
    dataFiles = glob.glob(directories["data"] + "*.hdf5")
    print(len(dataFiles))
    print(dataFiles)


def numpy_combine(args):
    np.set_printoptions(linewidth=np.inf)
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

    if args.wandb:
        initialize_wandb(
            args,
            timestamp,
            method="TT",
            tags=["PDEBench", "TT-ICE", MACHINE_ALIAS, args.M, args.type],
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
        len(train_simulations),
        len(val_simulations),
        len(test_simulations),
    )
    train_batch = []
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
        # train_batch.append(np.concatenate(sim,axis=-1).transpose(0,1,2,4,3)[...,None])
        sim = np.concatenate(sim, axis=-1)
        # print(sim.shape)
        for state_idx in range(num_states):
            sim[..., state_idx, :], normalizing_constant1, normalizing_constant2 = normalize(
                sim[..., state_idx, :], method=args.normalization
            )
            normalizing_constants_sim.append(
                np.array([normalizing_constant1, normalizing_constant2])[..., None]
            )
        # print(sim.mean(0).mean(0).mean(0).mean(-1))
        train_batch.append(sim[..., None])
        normalizing_constants_batch.append(np.concatenate(normalizing_constants_sim, axis=-1).T)
    # print(np.concatenate(normalizing_constants_sim,axis=-1).shape)
    # print(normalizing_constants_batch)

    # return
    train_batch = np.concatenate(train_batch, axis=-1)
    train_batch = train_batch.reshape(
        args.reshaping + [num_states, num_timesteps, args.batch_size], order=ORD
    )
    batch_norm = nla.norm(train_batch)

    dataset = dmt.ttObject(
        train_batch,
        epsilon=args.epsilon,
        keepData=False,
        samplesAlongLastDimension=True,
        method="ttsvd",
    )
    dataset.normalizing_constants = normalizing_constants_batch.copy()

    tic = time.time()
    dataset.ttDecomp(dtype=np.float64)
    batch_time = time.time() - tic
    total_time += batch_time

    rec = dataset.reconstruct(dataset.ttCores[-1][:, -args.batch_size :, 0]).squeeze(0)
    error_before_update = 0
    # print(rec.shape,train_batch.shape)
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
        # val_data = val_data.reshape(args.reshaping+[num_timesteps,num_states,1], order=ORD)
        val_data = val_data[..., None]
        val_data = val_data.reshape(args.reshaping + [num_states, num_timesteps, 1], order=ORD)
        rec = dataset.reconstruct(
            dataset.projectTensor(
                val_data,
            )
        ).squeeze(0)
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
        # test_data = np.concatenate(test_data,axis=-1).transpose(0,1,2,4,3)[...,None]
        test_data = np.concatenate(test_data, axis=-1)
        for state_idx in range(num_states):
            test_data[..., state_idx, :], _, _ = normalize(
                test_data[..., state_idx, :], method=args.normalization
            )
        # test_data = test_data.reshape(args.reshaping+[num_timesteps,num_states,1], order=ORD)
        test_data = test_data[..., None]
        test_data = test_data.reshape(args.reshaping + [num_states, num_timesteps, 1], order=ORD)
        rec = dataset.reconstruct(
            dataset.projectTensor(
                test_data,
            )
        ).squeeze(0)
        approximation_error = nla.norm(rec - test_data) / nla.norm(test_data)
        test_errors.append(approximation_error)

    ranks = dataset.ttRanks[1:-1].copy()
    # for core in dataset.transfer_nodes:
    #     ranks.append(core.shape[-1])
    # for leaf in dataset.leaves:
    #     ranks.append(leaf.shape[-1])
    print(
        f"{batch_index:04d} {dataset.ttCores[-1].shape[1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
    )

    # ADDED SAVING TO EXTRACT TURBULENCE SPECTRUM #
    # epsilon_str = "".join(f"{args.epsilon:0.2f}".split("."))
    dataset.seed = args.seed_idx

    if args.wandb:
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
        wandb.log(logging_dict)
    previous_ranks = dataset.ttRanks.copy()

    train_simulations = train_simulations[args.batch_size :]
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
            # train_batch.append(np.concatenate(sim,axis=-1).transpose(0,1,2,4,3)[...,None])
            sim = np.concatenate(sim, axis=-1)
            for state_idx in range(num_states):
                sim[..., state_idx, :], normalizing_constant1, normalizing_constant2 = normalize(
                    sim[..., state_idx, :], method=args.normalization
                )
                normalizing_constants_sim.append(
                    np.array([normalizing_constant1, normalizing_constant2])[..., None]
                )
            train_batch.append(sim[..., None])
            normalizing_constants_batch.append(np.concatenate(normalizing_constants_sim, axis=-1).T)
        train_batch = np.concatenate(train_batch, axis=-1)
        train_batch = train_batch.reshape(
            args.reshaping + [num_states, num_timesteps, args.batch_size], order=ORD
        )
        batch_norm = nla.norm(train_batch)
        # print(train_batch.shape,batch_norm,
        #   normalizing_constants_batch[-1]
        #   )
        projection = dataset.reconstruct(
            dataset.projectTensor(
                train_batch,
            )
        ).squeeze(0)
        error_before_update = nla.norm(projection - train_batch) / batch_norm
        # print(dataset.projectTensor(train_batch,).shape, nla.norm(projection),projection.shape)
        tic = time.time()
        dataset.ttICEstar(
            train_batch,
            heuristicsToUse=HEUR_2_USE,
            epsilon=args.epsilon,
            occupancyThreshold=OCCUPANCY,
        )
        batch_time = time.time() - tic
        # print(dataset.ttCores[-1].shape)
        dataset.normalizing_constants.extend(normalizing_constants_batch)
        rec = dataset.reconstruct(
            dataset.ttCores[-1][:, -train_batch.shape[-1] :, 0]
            # dataset.root.core[...,-args.batch_size:]
        ).squeeze(0)
        error_after_update = nla.norm(rec - train_batch) / batch_norm
        total_time += batch_time
        update_flag = dataset.ttRanks != previous_ranks
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
                rec = dataset.reconstruct(
                    dataset.projectTensor(
                        val_data,
                    )
                ).squeeze(0)
                # print(rec.shape,val_data.shape)
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
                # test_data = np.concatenate(test_data,axis=-1).transpose(0,1,2,4,3)[...,None]
                test_data = np.concatenate(test_data, axis=-1)
                for state_idx in range(num_states):
                    test_data[..., state_idx, :], _, _ = normalize(
                        test_data[..., state_idx, :], method=args.normalization
                    )
                test_data = test_data[..., None]
                test_data = test_data.reshape(
                    args.reshaping + [num_states, num_timesteps, 1], order=ORD
                )
                rec = dataset.reconstruct(
                    dataset.projectTensor(
                        test_data,
                    )
                ).squeeze(0)
                # print(rec.shape,test_data.shape)
                approximation_error = nla.norm(rec - test_data) / nla.norm(test_data)
                test_errors.append(approximation_error)

            ranks = dataset.ttRanks[1:-1].copy()
            # for core in dataset.transfer_nodes:
            #     ranks.append(core.shape[-1])
            # for leaf in dataset.leaves:
            #     ranks.append(leaf.shape[-1])
        previous_ranks = dataset.ttRanks.copy()
        print(
            f"{batch_index:04d} {dataset.ttCores[-1].shape[1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )
        if args.wandb:
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
            wandb.log(logging_dict)
        train_simulations = train_simulations[args.batch_size :]

        # ADDED SAVING TO EXTRACT TURBULENCE SPECTRUM #
        # epsilon_str = "".join(f"{args.epsilon:0.2f}".split("."))
        dataset.seed = args.seed_idx
        # dataset.saveData(f'PDEBench3_{args.normalization}_e{epsilon_str}',directory='/nfs/turbo/coe-goroda/doruk_temp/')
    # numel = np.prod(dataset.root.core.shape)
    # print(dataset.root.shape)
    # for core in dataset.transfer_nodes:
    #     print(core.shape)
    #     numel += np.prod(core.shape)
    # for leaf in dataset.leaves:
    #     print(leaf.shape)
    #     numel += np.prod(leaf.shape)
    # print(numel)
    # print(np.prod(train_batch.shape[:-1]),dataset.root.shape[-1])
    # print(dataset.batch_count)
    # print(dataset.original_shape)

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


def hdf5_combine(args):

    # args = get_args(args)
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # num_states = len(STATES)
    # data_loc = args.data_location + PATH_SEP + f"{args.type}" + PATH_SEP + f"{args.M}" + PATH_SEP
    # num_simulations = len(glob.glob(data_loc + "sim*_ts00.npy"))
    # num_timesteps = len(glob.glob(data_loc + "sim000_ts*.npy"))
    raise NotImplementedError("This method is not yet implemented")


def hdf5_nocombine(args):

    # args = get_args()
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
