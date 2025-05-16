#!.env/bin/python -u
import os
import glob
import time
import wandb
import argparse
import datetime

import numpy as np
import htucker as ht
import numpy.linalg as nla

from experiment_utils import initialize_wandb_catgel,get_args_catgel,normalize


MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7 ## Not used for this type of experiments
VAL_RATIO = 0.1 ## Not used for this type of experiments
TEST_RATIO = 0.2 ## Not used for this type of experiments
ORD = "F"
MACHINE_ALIAS = "LH"


def compress_catgel_full_sim_HT(args):
    # get data count
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if args.wandb:
        initialize_wandb_catgel(args,timestamp,tags=['CatGel','HTucker',MACHINE_ALIAS],method="HT")

    data_loc = args.data_location
    training_simulations = glob.glob(data_loc+f'/3rd/{args.type}/*.npy')
    training_simulations.sort()
    test_simulations = glob.glob(data_loc+f'/5th/{args.type}/*.npy')

    normalizing_constants = []
    training_snapshot = np.load(training_simulations[0])
    for state_idx in range(training_snapshot.shape[-1]):
        training_snapshot[..., state_idx], normalizing_constant1, normalizing_constant2 = normalize(training_snapshot[..., state_idx], method = args.normalization) 
        normalizing_constants.append(
            np.array([normalizing_constant1,normalizing_constant2])[..., None]
        )

    total_time = 0
    batch_index = 0
    # print(training_snapshot.transpose(0,1,2,4,3)[...,None].shape)

    training_snapshot = training_snapshot.transpose(0,1,2,4,3)[...,None]
    batch_norm = nla.norm(training_snapshot)

    batch_along = len(training_snapshot.shape)-1 # No -1 needed here since the batch dimension is created afterwards
    dataset = ht.HTucker()
    dataset.initialize(
        training_snapshot,
        batch = True,
        batch_dimension = batch_along
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
        training_snapshot,
        dimension_tree=dimension_tree,
        batch_dimension=batch_along,
    )
    batch_time = time.time()-tic
    total_time += batch_time

    rec = dataset.reconstruct(
            dataset.root.core[...,-args.batch_size:]
        )

    error_before_update = 0
    error_after_update = nla.norm(rec-training_snapshot)/batch_norm
    dataset.normalizing_constants = [np.concatenate(normalizing_constants,axis=-1).T] #type:ignore

    test_errors = []

    for simulation in test_simulations[:]:
        test_snapshot = np.load(simulation)
        for state_idx in range(test_snapshot.shape[-1]):
            test_snapshot[..., state_idx], _, _ = normalize(test_snapshot[..., state_idx], method = args.normalization) 
        test_snapshot = test_snapshot.transpose(0,1,2,4,3)[...,None]
        rec = dataset.reconstruct(
            dataset.project(
                test_snapshot,
                batch=True,
                batch_dimension=batch_along
            )
        )
        approximation_error = nla.norm(rec-test_snapshot)/nla.norm(test_snapshot)
        test_errors.append(approximation_error)

    ranks = []
    for core in dataset.transfer_nodes:
        ranks.append(core.shape[-1])
    for leaf in dataset.leaves:
        ranks.append(leaf.shape[-1])
    print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")

    if args.wandb:
        logging_dict = {
                "compression_ratio": dataset.compression_ratio,
                "error_before_update": error_before_update, 
                "error_after_update": error_after_update,
                "image_count": dataset.root.core.shape[-1],
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
            training_snapshot[..., state_idx], normalizing_constant1, normalizing_constant2 = normalize(training_snapshot[..., state_idx], method = args.normalization) 
            normalizing_constants.append(
                np.array([normalizing_constant1,normalizing_constant2])[..., None]
            )
        training_snapshot = training_snapshot.transpose(0,1,2,4,3)[...,None]
        batch_norm = nla.norm(training_snapshot)
        projection = dataset.reconstruct(
                dataset.project(
                training_snapshot,
                batch=True,
                batch_dimension=batch_along
            ))
        error_before_update = nla.norm(projection-training_snapshot)/batch_norm
        tic = time.time()
        update_flag = dataset.incremental_update_batch(
            training_snapshot,
            batch_dimension=batch_along,
            append=True,
        )
        batch_time = time.time()-tic
        rec = dataset.reconstruct(
            dataset.root.core[...,-training_snapshot.shape[-1]:]
            # dataset.root.core[...,-args.batch_size:]
        )
        error_after_update = nla.norm(rec-training_snapshot)/batch_norm
        total_time += batch_time

        # print(update_flag)
        if update_flag:
            for simulation in test_simulations[:]:
                test_snapshot = np.load(simulation)
                for state_idx in range(test_snapshot.shape[-1]):
                    test_snapshot[..., state_idx], _, _ = normalize(test_snapshot[..., state_idx], method = args.normalization) 
                test_snapshot = test_snapshot.transpose(0,1,2,4,3)[...,None]
                rec = dataset.reconstruct(
                    dataset.project(
                        test_snapshot,
                        batch=True,
                        batch_dimension=batch_along
                    )
                )
                approximation_error = nla.norm(rec-test_snapshot)/nla.norm(test_snapshot)
                test_errors.append(approximation_error)

            ranks = []
            for core in dataset.transfer_nodes:
                ranks.append(core.shape[-1])
            for leaf in dataset.leaves:
                ranks.append(leaf.shape[-1])
    
        print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):09.5f} {round(total_time, 5):10.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
        if args.wandb:
            logging_dict = {
                    "compression_ratio": dataset.compression_ratio,
                    "error_before_update": error_before_update, 
                    "error_after_update": error_after_update,
                    "image_count": dataset.root.core.shape[-1],
                    "batch_time": batch_time,
                    "total_time": total_time,
                    "test_error": np.mean(test_errors),
                    "batch_norm": batch_norm,
            }
            for idx, rank in enumerate(ranks):
                logging_dict[f"rank_{idx}"] = rank
            wandb.log(logging_dict)

    # print(len(test_errors),np.mean(test_errors))


if __name__ == '__main__':
    args = get_args_catgel()

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass

    # if args.reshaping == []:
    #     print("Reshaping is not provided, using baseline reshaping for Self-oscillating gel data.")
    #     args.reshaping = [2,4,4,4,8,8,2]
    
    # assert np.prod(args.resize) == np.prod(args.reshaping), "Reshaping and resizing do not match."
    # args.reshaping.extend([sum(args.states)])
    # print("Reshaping used: ", args.reshaping)


    print(args)
    overall_tic = time.time()
    compress_catgel_full_sim_HT(args)
    overall_time = time.time() - overall_tic
    print(f'Time for entire process (s): {round(overall_time,3)}')