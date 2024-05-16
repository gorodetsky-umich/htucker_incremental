#!.env/bin/python
import os
import glob
import time
import h5py
import wandb
import random
import argparse
import datetime

import numpy as np
import DaMAT as dmt
from tqdm import tqdm
import numpy.linalg as nla

from functools import reduce
from multiprocessing import Pool
from compress_PDEBenchHT import get_args

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"
HEUR_2_USE = ['skip', 'occupancy']

def main():
    args = get_args()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    if args.type in  ["rand", "random", "randomized", "Rand", "RAND", "Random"]:
        args.type = "Rand"
    elif args.type in ["turb", "turbulence", "turbulent", "Turb", "TURB", "Turbulence", "Turbulent"]:
        args.type = "Turb"
    else:
        raise ValueError("Type of simulation is not recognized.")
    print(args)

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
    else:
        seed_idx = args.seed_idx
    
    states = [
        'Vx', 'Vy', 'Vz', 'density', 'pressure'
    ]
    num_states = len(states)

    if args.type == "Rand":
        M = "0.1"
    elif args.type == "Turb":
        M = "1.0"
    else:
        raise ValueError("Type of simulation is not recognized.")
    
    file_name = f"3D_CFD_{args.type}_M{M}_Eta1e-08_Zeta1e-08_periodic_train.hdf5"
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
            name="PDEBench_HT_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
            config=run_config,
            tags=["PDEBench", "TT-ICE", "MBP23"],
        )

    # print(args.reshaping)
    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for PDEBench")
        if args.type == "Rand":
            # args.reshaping = [8,4,4,8,4,4,8,4,4]
            args.reshaping = [8,4,4,8,4,4,8,4,4]
        elif args.type == "Turb":
            # args.reshaping = [8,8,8,8,8,8]
            args.reshaping = [8,8,8,8,8,8]
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
            if os.path.exists(reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data"])):
                print("Data location is not provided, using default location")
                data_loc = reduce(os.path.join,[PATH_SEP]+HOME.split(os.path.sep)+["data"])
            else:
                raise IsADirectoryError("Please provide the data location")
    else: 
        data_loc = args.data_location
        # if args.numpy:
        #     assert glob.glob(args.data_location+"*.npy"), "There are no numpy files in the provided directory."

    directories = {
        "data" : data_loc+PATH_SEP,
        "cores" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["savedCores"])+PATH_SEP,
        "metric" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["experiments","BigEarthNet"])+PATH_SEP,
    }
    dataFiles = glob.glob(directories["data"]+"*.hdf5")
    print(len(dataFiles))
    print(dataFiles)
    




if __name__ == "__main__":
    main()