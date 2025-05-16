#!.env/bin/python -u
import os
import glob
import time
import copy
import h5py
import wandb
import random
import argparse
import datetime

import numpy as np
import htucker as ht
import numpy.linalg as nla

from functools import reduce, partial
from multiprocessing import Pool
from experiment_utils import normalize, get_args_pdebench
from compress_PDEBench_HT import read_snapshot

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 1
VAL_RATIO = 0.1
TEST_RATIO = 0.1
ORD = "F"
STATES = [
        'Vx', 'Vy', 'Vz', 'density', 'pressure'
    ]
MACHINE_ALIAS = "LH"
METRICS_FILE = "PDEBench_comparison_metrics_"

def parser_pdebench():
    parser = argparse.ArgumentParser(description='This script is to compare Hierarchical Tucker decomposition and Batch Hierarcical Tucker decomposition.')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, help='epsilon value', default=0.1)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('--states', dest='states', nargs='+', type=int, help='Determines the states that will be compressed', default=list(range(len(STATES))))
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    parser.add_argument('-t', '--type', dest='type', type=str, help='Type of simulation data', default="Rand")
    parser.add_argument('-c', '--combine', dest='combine', action='store_true', help='Combine timesteps of the simulation', default=False)
    parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Use extracted numpy files to read data' , default=False)
    parser.add_argument('-M', '--mach_number', dest='M', help='Mach number for the simulations', default=None)
    parser.add_argument('--method', dest='method', help='method of compression', default=None)
    parser.add_argument('-N', '--normalization', dest='normalization', type=str, help='Normalization method used', default='none')
    return parser.parse_args()

def initialize_wandb(args, timestamp, tags = []):
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
        name = f"PDEBench_{args.method}_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
        config = run_config,
        tags = tags,
    )

def hierarchical_tucker_pdebench(args):
    random.seed(args.seed_idx)
    np.random.seed(args.seed_idx)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    num_states = len(args.states)
    data_loc = args.data_location+PATH_SEP+f'{args.type}'+PATH_SEP+f'{args.M}'+PATH_SEP
    print(data_loc)
    num_simulations = len(glob.glob(data_loc+'sim*_ts00.npy'))
    num_timesteps = len(glob.glob(data_loc+'sim000_ts*.npy'))
    grid_shape = list(np.load(data_loc+'sim000_ts00.npy').shape[:-1])
    print(num_simulations,num_timesteps,grid_shape)
    assert np.prod(grid_shape) == np.prod(args.reshaping), "Reshaping does not match the shape of the data"

    if args.wandb:
        initialize_wandb(args, timestamp, tags = ["PDEBench", "HTvsBHT", args.method, MACHINE_ALIAS, args.M, args.type])

    train_simulations = random.sample(range(0, num_simulations), int(num_simulations*TRAIN_RATIO))
    # val_simulations = random.sample(list(set(range(0, num_simulations))-set(train_simulations)), int(num_simulations*VAL_RATIO))
    # test_simulations = list(set(range(0, num_simulations))-set(train_simulations)-set(val_simulations))
    
    for k in range(len(train_simulations)//args.batch_size):
        ## Preallocate the array
        train_batch = np.zeros([64,64,64] +[num_states, 21 , args.batch_size*(k+1)])
        # print(train_batch.shape)
        simulationwise_sim_norm = np.zeros((k+1)*args.batch_size)
        for simulation_idx, simulation in enumerate(train_simulations[:(k+1)*args.batch_size]):
            with Pool() as p:
                sim = p.map(partial(read_snapshot,data_loc=data_loc,simulation=simulation,states=args.states),list(range(num_timesteps)))
            sim = np.concatenate(sim,axis=-1)
            for state_idx in range(num_states):
                # print(k,simulation_idx,state_idx)
                train_batch[...,state_idx,:,simulation_idx], normalizing_constant1, normalizing_constant2 = normalize(sim[...,state_idx,:],method = args.normalization)
            simulationwise_sim_norm[simulation_idx] = nla.norm(train_batch[...,simulation_idx])
        # print(train_batch.shape)
        # train_batch = train_batch.reshape(args.reshaping+[num_timesteps,num_states,(k+1)*args.batch_size], order=ORD)
        train_batch = train_batch.reshape(args.reshaping+[num_states, num_timesteps,(k+1)*args.batch_size], order=ORD)
        # print(train_batch.shape)
        # quit()
        batch_norm = nla.norm(train_batch)
        # batch_along = len(train_batch.shape)-1
        dataset = ht.HTucker()
        dataset.initialize(
            train_batch,
        )
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
        
        simulationwise_error = np.zeros((k+1)*args.batch_size)
        simulationwise_approx_norm = np.zeros((k+1)*args.batch_size)
        for simulation_idx, simulation in enumerate(train_simulations[:(k+1)*args.batch_size]):
            temp_ht = copy.deepcopy(dataset)
            temp_ht.leaves[-1].core = temp_ht.leaves[-1].core[simulation_idx,:][None,...]
            temp_ht.reconstruct_all()
            simulationwise_error[simulation_idx] = nla.norm(train_batch[...,simulation_idx]-temp_ht.root.core.squeeze())/simulationwise_sim_norm[simulation_idx]
            simulationwise_approx_norm[simulation_idx] = nla.norm(temp_ht.root.core)

        ranks = []
        for core in dataset.transfer_nodes:
            ranks.append(core.shape[-1])
        for leaf in dataset.leaves:
            ranks.append(leaf.shape[-1])
        

        line_to_append = []
        line_to_append.append(f"{(k+1):3d}")
        line_to_append.append(f"{(k+1)*args.batch_size:6d}")
        line_to_append.append(f"{round(compression_time, 5):11.5f}")
        line_to_append.append(f"{batch_norm:20.3f}")
        line_to_append.append(f"{round(np.mean(simulationwise_error), 5):0.5f}")
        line_to_append.append(f"{round(np.sqrt(np.square(simulationwise_error).sum()),5):0.5f}")
        line_to_append.append(f"{round(np.sqrt(1-(np.square(simulationwise_approx_norm).sum()/np.square(simulationwise_sim_norm).sum())),5):0.5f}")
        line_to_append.append(f"{round(dataset.compression_ratio, 5):8.3f}")
        line_to_append.append(" ".join(map(lambda x: f'{x:4d}', ranks)))
        print(" ".join(line_to_append))
        with open(DIRECTORIES["metric"]+METRICS_FILE+f"eps_{args.epsilon}_{args.method}.txt", 'a') as f:
            f.writelines(" ".join(map(str,line_to_append))+"\n")
        
        logging_dict = {
            "compression_ratio": dataset.compression_ratio,
            # "error_before_update": error_before_update, 
            # "error_after_update": error_after_update,
            "image_count": (k+1)*args.batch_size,
            # "mean_error": np.mean(simulationwise_error),
            "actual_error": np.sqrt(1-(np.square(simulationwise_approx_norm).sum()/np.square(simulationwise_sim_norm).sum())),
            # "batch_time": batch_time,
            "total_time": compression_time,
            # "val_error": np.mean(val_errors),
            # "test_error": np.mean(test_errors),
            "batch_norm": batch_norm,
        }

        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank
        if args.wandb:
            wandb.log(logging_dict)

def batch_hierarchical_tucker_pdebench(args):
    random.seed(args.seed_idx)
    np.random.seed(args.seed_idx)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    num_states = len(args.states)
    data_loc = args.data_location+PATH_SEP+f'{args.type}'+PATH_SEP+f'{args.M}'+PATH_SEP
    print(data_loc)
    num_simulations = len(glob.glob(data_loc+'sim*_ts00.npy'))
    num_timesteps = len(glob.glob(data_loc+'sim000_ts*.npy'))
    grid_shape = list(np.load(data_loc+'sim000_ts00.npy').shape[:-1])
    print(num_simulations,num_timesteps,grid_shape)
    assert np.prod(grid_shape) == np.prod(args.reshaping), "Reshaping does not match the shape of the data"

    if args.wandb:
        initialize_wandb(args, timestamp, tags = ["PDEBench", "HTvsBHT", args.method, MACHINE_ALIAS, args.M, args.type])

    train_simulations = random.sample(range(0, num_simulations), int(num_simulations*TRAIN_RATIO))
    # val_simulations = random.sample(list(set(range(0, num_simulations))-set(train_simulations)), int(num_simulations*VAL_RATIO))
    # test_simulations = list(set(range(0, num_simulations))-set(train_simulations)-set(val_simulations))
    
    for k in range(len(train_simulations)//args.batch_size):
        ## Preallocate the array
        train_batch = np.zeros([64,64,64] +[num_states, 21 ,args.batch_size*(k+1)])
        # print(train_batch.shape)
        simulationwise_sim_norm = np.zeros((k+1)*args.batch_size)
        for simulation_idx, simulation in enumerate(train_simulations[:(k+1)*args.batch_size]):
            # print(simulation_idx)
            with Pool() as p:
                sim = p.map(partial(read_snapshot,data_loc=data_loc,simulation=simulation,states=args.states),list(range(num_timesteps)))
            sim = np.concatenate(sim,axis=-1)
            for state_idx in range(num_states):
                # print(k,simulation_idx,state_idx)
                train_batch[...,state_idx,:,simulation_idx], normalizing_constant1, normalizing_constant2 = normalize(sim[...,state_idx,:],method = args.normalization)
            simulationwise_sim_norm[simulation_idx] = nla.norm(train_batch[...,simulation_idx])
        # print(train_batch.shape)

        # train_batch = train_batch.reshape(args.reshaping+[num_timesteps,num_states,(k+1)*args.batch_size], order=ORD)
        train_batch = train_batch.reshape(args.reshaping+[num_states, num_timesteps,(k+1)*args.batch_size], order=ORD)
        # print(train_batch.shape)
        # quit()
        batch_norm = nla.norm(train_batch)
        batch_along = len(train_batch.shape)-1
        dataset = ht.HTucker()
        dataset.initialize(
            train_batch,
            batch=True,
            batch_dimension=batch_along
        )
        dimension_tree = ht.createDimensionTree(
        dataset.original_shape[:batch_along]+dataset.original_shape[batch_along+1:],
        numSplits = 2,
        minSplitSize = 1,
        )
        dimension_tree.get_items_from_level()
        dataset.rtol= args.epsilon

        tic = time.time()
        dataset.compress_leaf2root_batch(
            train_batch,
            dimension_tree=dimension_tree,
            batch_dimension=batch_along,
        )
        compression_time = time.time()-tic

        ranks = []
        for core in dataset.transfer_nodes:
            ranks.append(core.shape[-1])
        for leaf in dataset.leaves:
            ranks.append(leaf.shape[-1])

        simulationwise_approx_norm = np.zeros(dataset.root.core.shape[-1])
        for simulation_idx in range(dataset.root.core.shape[-1]):
            simulationwise_approx_norm[simulation_idx] = nla.norm(dataset.root.core[...,simulation_idx])


        approx_norm = nla.norm(dataset.root.core)

        simulationwise_error = np.sqrt(1-(np.square(simulationwise_approx_norm)/np.square(simulationwise_sim_norm)))
        rel_err = np.sqrt(1-(np.square(approx_norm)/np.square(batch_norm)))

        line_to_append = []
        line_to_append.append(f"{k+1:3d}")
        line_to_append.append(f"{(k+1)*args.batch_size:6d}")
        line_to_append.append(f"{round(compression_time, 5):11.5f}")
        line_to_append.append(f"{batch_norm:20.3f}")
        line_to_append.append(f"{round(np.mean(simulationwise_error),5):0.5f}")
        line_to_append.append(f"{round(rel_err,5):0.5f}")
        line_to_append.append(f"{round(dataset.compression_ratio, 5):8.3f}")
        line_to_append.append(" ".join(map(lambda x: f'{x:4d}', ranks)))
        print(" ".join(line_to_append))
        with open(DIRECTORIES["metric"]+METRICS_FILE+f"eps_{args.epsilon}_{args.method}.txt", 'a') as f:
            f.writelines(" ".join(map(str,line_to_append))+"\n")


        logging_dict = {
            "compression_ratio": dataset.compression_ratio,
            "image_count": (k+1)*args.batch_size,
            "mean_error": np.mean(simulationwise_error),
            "actual_error": rel_err,
            "total_time": compression_time,
            "batch_norm": batch_norm,
        }
        
        for idx, rank in enumerate(ranks):
            logging_dict[f"rank_{idx}"] = rank
        if args.wandb:
            wandb.log(logging_dict)

def hierarchical_tucker_pdebench_not_combined(args):
    raise NotImplementedError

def batch_hierarchical_tucker_pdebench_not_combined(args):
    raise NotImplementedError


if __name__ == "__main__":
    overall_start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    args = parser_pdebench()

    if args.type in  ["rand", "random", "randomized", "Rand", "RAND", "Random"]:
        args.type = "Rand"
    elif args.type in ["turb", "turbulence", "turbulent", "Turb", "TURB", "Turbulence", "Turbulent"]:
        args.type = "Turb"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if (args.type == "Rand") and (args.M is None):
        print(f'No M is given. Using default M=1.0')
        args.M = "1.0"
    elif (args.type == "Turb") and (args.M is None):
        args.M = "1.0"
    else:
        raise ValueError("Type of simulation is not recognized.")

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        seed_idx = int(rng.integers(MAX_SEED))
        args.seed_idx = seed_idx
    else:
        seed_idx = args.seed_idx

    random.seed(seed_idx)  # Fix the seed to a random number
    np.random.seed(seed_idx)  # Fix the seed to a random number

    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for PDEBench")
        if args.type == "Rand":
            args.reshaping = [8,4,4,8,4,4,8,4,4]
        elif args.type == "Turb":
            args.reshaping = [8,8,8,8,8,8]

    DIRECTORIES = {
        "data" : args.data_location+PATH_SEP,
        "cores" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["savedCores"])+PATH_SEP,
        "metric" : reduce(os.path.join,[PATH_SEP]+CWD.split(os.path.sep)+["experiments","PDEBench"])+PATH_SEP,
    }

    if not os.path.exists(DIRECTORIES["metric"]):
        os.makedirs(DIRECTORIES["metric"])

    print(args)
    overall_tic = time.time()
    if (args.method == 'ht') and args.combine:
        print("HT, combining timesteps")
        hierarchical_tucker_pdebench(args)
    elif (args.method == 'ht'):
        print("HT, not combining timesteps")
        hierarchical_tucker_pdebench_not_combined(args)
    elif (args.method == 'bht') and args.combine:
        print("BHT, combining timesteps")
        batch_hierarchical_tucker_pdebench(args)
    elif (args.method == 'bht'): 
        print("BHT, not combining timesteps")
        batch_hierarchical_tucker_pdebench_not_combined(args)
    else:
        raise NotImplementedError
    
    overall_time = time.time() - overall_tic
    print(f'Time for entire process (s): {round(overall_time,3)}')

    