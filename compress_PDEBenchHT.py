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
import htucker as ht
from tqdm import tqdm
import numpy.linalg as nla

from functools import reduce
from multiprocessing import Pool

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"

__all__ = [
    "get_args",
]

def read_simulations(simulation_idx, states, timesteps = None, path_to_data ="./", file_name=""):
    sim = []
    # print(path_to_data,file_name, simulation_idx, timesteps)
    with h5py.File(os.path.join(path_to_data, file_name), 'r') as f:
        for state in states:
            # print(state, f[state].shape)
            if timesteps is None:
                ## Read all timesteps
                sim.append(f[state][simulation_idx,:,...].transpose(1,2,3,0)[...,None])
            else:
                ## Read only the specified timestep(s)
                sim.append(f[state][simulation_idx,timesteps,...][...,None])
    
    return np.concatenate(sim,axis=-1)[...,None]

def get_args():
    parser = argparse.ArgumentParser(description='This script reads the PDEBench simulation snapshots and compresses them using the HT format.')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', help='epsilon value', default=0.1)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    parser.add_argument('-t', '--type', dest='type', type=str, help='Type of simulation data', default="Rand")
    parser.add_argument('-c', '--combine', dest='combine', action='store_true', help='Combine timesteps of the simulation', default=False)

    return parser.parse_args()

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
            tags=["PDEBench", "HTucker", "MBP23"],
        )
    print(args.reshaping)
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

    with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
        for state in states:
            print(state, f[state].shape)
        num_simulations = f[states[0]].shape[0]
        num_time_steps = f[states[0]].shape[1]
        grid_shape = list(f[states[0]].shape[2:])
    

    anan = read_simulations(0, states, path_to_data=directories["data"], file_name=file_name)#.transpose(2,3,4,0,1)
    

    print(anan.shape)
    # yarrak = [
    #     (0, states, 0, directories["data"], file_name),
    #     (0, states, 1, directories["data"], file_name),
    #     (0, states, 2, directories["data"], file_name),
    #     (0, states, 3, directories["data"], file_name),
    #     ]
    # yarrak = [(ii, states ,jj , directories["data"], file_name) for ii in range(args.batch_size) for jj in range(num_time_steps)]
    # print(yarrak)
    # yarrak = [(i, states, None, directories["data"], file_name) for i in range(args.batch_size)]
    
    # tt= time.time()
    # with Pool() as pool:
    #     deneme123 = pool.starmap(read_simulations, yarrak)
    #     # deneme123 = pool.starmap(read_simulations, yarrak)
    # print(np.concatenate(deneme123,axis=-1).shape)
    # print(time.time()-tt)
    # # print(np.array(deneme123).transpose(3,4,5,2,1,0).shape)

    # tt= time.time()
    # batch_shape = grid_shape+[num_time_steps,num_states,args.batch_size]
    # train_batch = np.zeros(batch_shape)
    # for ii in range(args.batch_size):
    #     sim_idx = ii
    #     # print(ii, sim_idx)
    #     with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
    #         for state_idx, state in enumerate(states):
    #             train_batch[...,state_idx,ii] = f[state][sim_idx].transpose(1,2,3,0)
    # print(train_batch.shape)
    # print(time.time()-tt)

    # sim_idx = np.array([ii for ii in range(args.batch_size)])
    # tt= time.time()
    # with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
    #     for state_idx, state in enumerate(states):
    #         # print(f[state][sim_idx,...].shape)
    #         train_batch[...,state_idx,:] = f[state][sim_idx,...].transpose(2,3,4,1,0)
    # print(train_batch.shape)
    # print(time.time()-tt)




    # anan = [1,2,3,4,5,6,7,8,9,10]
    # print(anan)
    # for i in anan[:args.batch_size]:
    #     print(i)
    #     anan.remove(i)
    # print(anan)

    # return
    
    print(state)
    assert np.prod(grid_shape) == np.prod(args.reshaping), "Reshaping does not match the shape of the data"
    
    train_simulations = random.sample(range(0, num_simulations), int(num_simulations*TRAIN_RATIO))
    val_simulations = random.sample(list(set(range(0, num_simulations))-set(train_simulations)), int(num_simulations*VAL_RATIO))
    test_simulations = list(set(range(0, num_simulations))-set(train_simulations)-set(val_simulations))
    total_time = 0
    batch_index = 0

    if args.combine:
        
        print(grid_shape, num_time_steps, num_states, num_simulations)
        batch_shape = grid_shape+[num_time_steps,num_states,args.batch_size]

        train_batch = np.zeros(batch_shape)
        print(train_batch.shape)

        
        sim_idx = np.array(
            [ii for ii in train_simulations[:args.batch_size]]
        )
        train_simulations = train_simulations[args.batch_size:]
        sim_idx.sort()
        # tt= time.time()
        with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
            for state_idx, state in enumerate(states):
                # print(f[state][sim_idx,...].shape)
                train_batch[...,state_idx,:] = f[state][sim_idx,...].transpose(2,3,4,1,0)

        # for ii in range(args.batch_size):
        #     sim_idx = train_simulations.pop(0)
        #     print(ii, sim_idx)
        #     with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
        #         for state_idx, state in enumerate(states):
        #             train_batch[...,state_idx,ii] = f[state][sim_idx].transpose(1,2,3,0)
        # print(train_batch.shape)
        train_batch = train_batch.reshape(args.reshaping+[num_time_steps,num_states,args.batch_size], order=ORD)
        batch_norm = nla.norm(train_batch)
        print(train_batch.shape)

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
        batch_time = time.time()-tic
        total_time += batch_time

        rec = dataset.reconstruct(
            dataset.root.core[...,-args.batch_size:]
        )
        error_before_update = 0
        error_after_update = nla.norm(rec-train_batch)/batch_norm

        val_errors = []
        test_errors = []
        with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
            for sim_idx in val_simulations:
                val_data = np.zeros((grid_shape+[num_time_steps,num_states,1]))
                for state_idx, state in enumerate(states):
                    val_data[...,state_idx,:] = f[state][sim_idx,...][None,...].transpose(2,3,4,1,0)
                val_data = val_data.reshape(args.reshaping+[num_time_steps,num_states,1], order=ORD)
                rec = dataset.reconstruct(
                    dataset.project(
                        val_data,
                        batch=True,
                        batch_dimension=batch_along
                    )
                )
                approximation_error = nla.norm(rec-val_data)/nla.norm(val_data)
                val_errors.append(approximation_error)

            for sim_idx in test_simulations:
                test_data = np.zeros((grid_shape+[num_time_steps,num_states,1]))
                for state_idx, state in enumerate(states):
                    test_data[...,state_idx,:] = f[state][sim_idx,...][None,...].transpose(2,3,4,1,0)
                test_data = test_data.reshape(args.reshaping+[num_time_steps,num_states,1], order=ORD)
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
        
        print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
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

        if args.wandb:
            wandb.log(logging_dict)

        while train_simulations:
            batch_index += 1
            train_batch = np.zeros(batch_shape)
            sim_idx = np.array(
                [ii for ii in train_simulations[:args.batch_size]]
            )
            sim_idx.sort()
            train_simulations = train_simulations[args.batch_size:]
            with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
                for state_idx, state in enumerate(states):
                    # print(f[state][sim_idx,...].shape)
                    train_batch[...,state_idx,:] = f[state][sim_idx,...].transpose(2,3,4,1,0)
            train_batch = train_batch.reshape(args.reshaping+[num_time_steps,num_states,args.batch_size], order=ORD)
            batch_norm = nla.norm(train_batch)
            print(train_batch.shape)

            # for ii in range(args.batch_size):
            #     sim_idx = train_simulations.pop(0)
            #     with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
            #         for state_idx, state in enumerate(states):
            #             train_batch[...,state_idx,ii] = f[state][sim_idx].transpose(1,2,3,0)
            # train_batch = train_batch.reshape(args.reshaping+[num_time_steps,num_states,args.batch_size], order=ORD)
            # batch_norm = nla.norm(train_batch)
            projection = dataset.reconstruct(
                dataset.project(
                train_batch,
                batch=True,
                batch_dimension=batch_along
            ))
            error_before_update = nla.norm(projection-train_batch)/batch_norm
            tic = time.time()
            update_flag = dataset.incremental_update_batch(
                train_batch,
                batch_dimension=batch_along,
                append=True,
            )
            batch_time = time.time()-tic
            rec = dataset.reconstruct(
                dataset.root.core[...,-train_batch.shape[-1]:]
                # dataset.root.core[...,-args.batch_size:]
            )
            error_after_update = nla.norm(rec-train_batch)/batch_norm
            total_time += batch_time
            if update_flag:
                val_errors = []
                test_errors = []
                with h5py.File(os.path.join(directories["data"], file_name), 'r') as f:
                    for sim_idx in val_simulations:
                        val_data = np.zeros((grid_shape+[num_time_steps,num_states,1]))
                        for state_idx, state in enumerate(states):
                            val_data[...,state_idx,:] = f[state][sim_idx,...][None,...].transpose(2,3,4,1,0)
                        val_data = val_data.reshape(args.reshaping+[num_time_steps,num_states,1], order=ORD)        
                        rec = dataset.reconstruct(
                            dataset.project(
                                val_data,
                                batch=True,
                                batch_dimension=batch_along
                            )
                        )
                        approximation_error = nla.norm(rec-val_data)/nla.norm(val_data)
                        val_errors.append(approximation_error)

                    for sim_idx in test_simulations:
                        test_data = np.zeros((grid_shape+[num_time_steps,num_states,1]))
                        for state_idx, state in enumerate(states):
                            test_data[...,state_idx,:] = f[state][sim_idx,...][None,...].transpose(2,3,4,1,0)
                        test_data = test_data.reshape(args.reshaping+[num_time_steps,num_states,1], order=ORD)        
                        rec = dataset.reconstruct(
                            dataset.project(
                                test_data,
                                batch=True,
                                batch_dimension=batch_along
                            )
                        )
                        approximation_error = nla.norm(rec-test_data)/nla.norm(test_data)
                        test_errors.append(approximation_error) 
                # val_errors = []
                # for image in val_images:
                #     val_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
                #     rec = dataset.reconstruct(
                #         dataset.project(
                #             val_data,
                #             batch=True,
                #             batch_dimension=batch_along
                #         )
                #     )
                #     approximation_error = nla.norm(rec-val_data)/nla.norm(val_data)
                #     val_errors.append(approximation_error)
                # test_errors = []
                # for image in test_images:
                #     test_data = np.load(image)[...,filter].reshape(args.reshaping)[...,None]
                #     rec = dataset.reconstruct(
                #         dataset.project(
                #             test_data,
                #             batch=True,
                #             batch_dimension=batch_along
                #         )
                #     )
                #     approximation_error = nla.norm(rec-test_data)/nla.norm(test_data)
                #     test_errors.append(approximation_error)
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
                "batch_norm": batch_norm,
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            if args.wandb:
                wandb.log(logging_dict)

            print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")

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
        temp_list = [(sim_idx, states , ts , directories["data"], file_name) for sim_idx in train_simulations for ts in range(num_time_steps)]
        train_simulations = temp_list.copy()

        temp_list = [(sim_idx, states , ts , directories["data"], file_name) for sim_idx in val_simulations for ts in range(num_time_steps)]
        val_simulations = temp_list.copy()

        temp_list = [(sim_idx, states , ts , directories["data"], file_name) for sim_idx in test_simulations for ts in range(num_time_steps)]
        test_simulations = temp_list.copy()



        raise NotImplementedError("Not combining timesteps is not implemented yet.")


if __name__ == "__main__":
    main()