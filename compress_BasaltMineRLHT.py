#!.env/bin/python
import os
import sys 
import cv2 
import glob
import time
import wandb
import random
import datetime
import argparse

import numpy as np
import htucker as ht
import numpy.linalg as nla

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
ORD = "F"
DEFAULT_RESIZE = [128, 128]

__all__ = [
    "get_args",
    "initialize_wandb",
]

def initialize_wandb(args,timestamp,tags,method):
    run_config = wandb.config = {
        "epsilon": args.epsilon,
        "batch_size": args.batch_size,
        "reshaping": args.reshaping,
        "frame_size": args.resize,
        "seed_idx": args.seed_idx,
    }
    wandb.init(
        project="HierarchicalTucker_experiments",
        name=f"BasaltMineRL_{method}_eps_"+"".join(f"{args.epsilon:0.2f}_".split("."))+f"batchsize_{args.batch_size:04d}_framesize_"+"_".join(map(str,args.resize))+"_"+"shape_"+"_".join(map(str,args.reshaping))+"_date_"+timestamp,
        config=run_config,
        tags=tags,
    )

def get_args():
    parser = argparse.ArgumentParser(description='This script reads the PDEBench simulation snapshots and compresses them using the HT format.')
    parser.add_argument('-s', '--seed', dest='seed_idx', type=int , help='Variable to pass seed index', default=None)
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float ,help='epsilon value', default=0.1)
    parser.add_argument('-d', '--data_location', dest='data_location', help='path to data', default=None)
    parser.add_argument('-r', '--reshaping', dest='reshaping', nargs='+', type=int, help='Determines the reshaping for the tensor stream', default=[])
    parser.add_argument('-z', '--resize', dest='resize', nargs='+', type=int, help='', default=None)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, help='Batch size', default=1)
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', help='Use wandb for logging', default=False)
    # parser.add_argument('-t', '--type', dest='type', type=str, help='Type of simulation data', default="Rand")
    # parser.add_argument('-c', '--combine', dest='combine', action='store_true', help='Combine timesteps of the simulation', default=False)
    # parser.add_argument('-n', '--numpy', dest='numpy', action='store_true', help='Use extracted numpy files to read data' , default=False)
    # parser.add_argument('-M', '--mach_number', dest='M', help='Mach number for the simulations', default=None)
    return parser.parse_args()


def main(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    tags = ['BasaltMineRL', 'HTucker', 'MBP23']

    if args.wandb:
        initialize_wandb(args,timestamp,tags,method="HT")
    
    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number

    video_files = glob.glob(os.path.join(args.data_location, '*.mp4'))
    total_time = 0
    frame_counter = 0
    video_counter = 0 
    total_frame_counter = 0
    resize_reshape = args.reshaping + [3,-1] # 3 for RGB channels

    train_videos = random.sample(video_files, int(TRAIN_RATIO*len(video_files)))
    val_videos = random.sample(list(set(video_files) - set(train_videos)), int(VAL_RATIO*len(video_files)))
    test_videos = list(set(video_files) - set(train_videos) - set(val_videos))
    print(resize_reshape)
    print(len(train_videos), len(val_videos), len(test_videos))
    frames = []
    train_video = cv2.VideoCapture(train_videos[0])
    for _ in range(args.batch_size):
        success_train, frame = train_video.read()
        if not success_train:
            break
        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
        frames.append(frame)
    train_batch = np.concatenate(frames,axis=-1)
    print(train_batch.shape)
    train_batch = train_batch.reshape(resize_reshape, order=ORD)
    print(train_batch.shape)
    batch_norm = nla.norm(train_batch)
    error_before_update = 0
    batch_along = len(train_batch.shape) - 1 # last dimension is the batch dimension
    dataset = ht.HTucker()
    dataset.initialize(
        train_batch,
        batch= True,
        batch_dimension= batch_along,
    )
    dim_tree = ht.createDimensionTree(
    dataset.original_shape[:batch_along]+dataset.original_shape[batch_along+1:],
    numSplits = 2,
    minSplitSize = 1,
)
    dim_tree.get_items_from_level()
    print(type(args.epsilon))
    dataset.rtol = args.epsilon
    tic = time.time()
    dataset.compress_leaf2root_batch(
        train_batch,
        dimension_tree = dim_tree,
        batch_dimension = batch_along,
    )
    batch_time = time.time()-tic
    total_time += batch_time
    batch_index = 0
    rec = dataset.reconstruct(
            dataset.root.core[...,-args.batch_size:]
        )
    frame_counter += args.batch_size
    total_frame_counter += args.batch_size
    error_after_update = nla.norm(rec-train_batch)/batch_norm
    val_errors = []
    test_errors = []

    # Projection error for validation videos
    for video_file in val_videos:
        video = cv2.VideoCapture(video_file)
        frames = []
        while True:
            success, frame = video.read()
            if not success:
                break
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
            frames.append(frame)
        val_batch = np.concatenate(frames,axis=-1)
        val_batch = val_batch.reshape(resize_reshape, order=ORD)
        rec = dataset.reconstruct(
            dataset.project(val_batch,batch=True,batch_dimension=batch_along),
            batch=True,
            )
        error = val_batch-rec
        elementwise_norm = nla.norm(val_batch,axis=0)
        error_norm = nla.norm(error,axis=0)
        for _ in range(len(error.shape)-2):
            elementwise_norm = nla.norm(elementwise_norm,axis=0)
            error_norm = nla.norm(error_norm,axis=0)
        val_errors.extend(error_norm/(elementwise_norm).tolist())

    # Projection error for test videos
    for video_file in test_videos:
        video = cv2.VideoCapture(video_file)
        frames = []
        while True:
            success, frame = video.read()
            if not success:
                break
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
            frames.append(frame)
        test_batch = np.concatenate(frames,axis=-1)
        test_batch = test_batch.reshape(resize_reshape, order=ORD)
        rec = dataset.reconstruct(
            dataset.project(test_batch,batch=True,batch_dimension=batch_along),
            batch=True,
            )
        error = test_batch-rec
        elementwise_norm = nla.norm(test_batch,axis=0)
        error_norm = nla.norm(error,axis=0)
        for _ in range(len(error.shape)-2):
            elementwise_norm = nla.norm(elementwise_norm,axis=0)
            error_norm = nla.norm(error_norm,axis=0)
        test_errors.extend((error_norm/elementwise_norm).tolist())
    # return
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
    print(f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
    batch_index += 1
    # print("Finished first step, entering incremental part")
    while success_train:
        frames = []
        for _ in range(args.batch_size):
            success_train, frame = train_video.read()
            if not success_train:
                break
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
            frames.append(frame)
        try:
            train_batch = np.concatenate(frames,axis=-1)
        except:
            break
        train_batch = train_batch.reshape(resize_reshape, order=ORD)
        batch_norm = nla.norm(train_batch)
        projection = dataset.reconstruct(
            dataset.project(train_batch,batch=True,batch_dimension=batch_along),
            batch=True,
            )
        error_before_update = nla.norm(projection-train_batch)/batch_norm
        tic = time.time()
        update_flag = dataset.incremental_update_batch(
            train_batch,
            batch_dimension = batch_along,
            append = True,
        )
        batch_time = time.time()-tic
        total_time += batch_time
        rec= dataset.reconstruct(
            dataset.project(train_batch,batch=True,batch_dimension=batch_along),
            batch=True,
            )
        error_after_update = nla.norm(rec-train_batch)/batch_norm
        frame_counter += train_batch.shape[-1]
        total_frame_counter += train_batch.shape[-1]
        if update_flag:
            for video_file in val_videos:
                video = cv2.VideoCapture(video_file)
                frames = []
                while True:
                    success, frame = video.read()
                    if not success:
                        break
                    frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
                    frames.append(frame)
                val_batch = np.concatenate(frames,axis=-1)
                val_batch = val_batch.reshape(resize_reshape, order=ORD)
                rec = dataset.reconstruct(
                    dataset.project(val_batch,batch=True,batch_dimension=batch_along),
                    batch=True,
                    )
                error = val_batch-rec
                elementwise_norm = nla.norm(val_batch,axis=0)
                error_norm = nla.norm(error,axis=0)
                for _ in range(len(error.shape)-2):
                    elementwise_norm = nla.norm(elementwise_norm,axis=0)
                    error_norm = nla.norm(error_norm,axis=0)
                val_errors.extend((error_norm/elementwise_norm).tolist())

            # Projection error for test videos
            for video_file in test_videos:
                video = cv2.VideoCapture(video_file)
                frames = []
                while True:
                    success, frame = video.read()
                    if not success:
                        break
                    frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
                    frames.append(frame)
                test_batch = np.concatenate(frames,axis=-1)
                test_batch = test_batch.reshape(resize_reshape, order=ORD)
                rec = dataset.reconstruct(
                    dataset.project(test_batch,batch=True,batch_dimension=batch_along),
                    batch=True,
                    )
                error = test_batch-rec
                elementwise_norm = nla.norm(test_batch,axis=0)
                error_norm = nla.norm(error,axis=0)
                for _ in range(len(error.shape)-2):
                    elementwise_norm = nla.norm(elementwise_norm,axis=0)
                    error_norm = nla.norm(error_norm,axis=0)
                test_errors.extend((error_norm/elementwise_norm).tolist())
            
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

        print(f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
        # print(f"{batch_index:04d} {dataset.root.core.shape[-1]:06d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
        batch_index += 1
    
    for train_video in train_videos[1:]:
        video_counter += 1
        frame_counter = 0
        train_video = cv2.VideoCapture(train_video)
        success_train = True
        while success_train:
            frames = []
            for _ in range(args.batch_size):
                success_train, frame = train_video.read()
                if not success_train:
                    break
                frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
                frames.append(frame)
            train_batch = np.concatenate(frames,axis=-1)
            train_batch = train_batch.reshape(resize_reshape, order=ORD)
            batch_norm = nla.norm(train_batch)
            projection = dataset.reconstruct(
                dataset.project(train_batch,batch=True,batch_dimension=batch_along),
                batch=True,
                )
            error_before_update = nla.norm(projection-train_batch)/batch_norm
            tic = time.time()
            update_flag = dataset.incremental_update_batch(
                train_batch,
                batch_dimension = batch_along,
                append = True,
            )
            batch_time = time.time()-tic
            total_time += batch_time
            rec= dataset.reconstruct(
                dataset.project(train_batch,batch=True,batch_dimension=batch_along),
                batch=True,
                )
            error_after_update = nla.norm(rec-train_batch)/batch_norm
            frame_counter += train_batch.shape[-1]
            total_frame_counter += train_batch.shape[-1]
            if update_flag:
                for video_file in val_videos:
                    video = cv2.VideoCapture(video_file)
                    frames = []
                    while True:
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
                        frames.append(frame)
                    val_batch = np.concatenate(frames,axis=-1)
                    val_batch = val_batch.reshape(resize_reshape, order=ORD)
                    rec = dataset.reconstruct(
                        dataset.project(val_batch,batch=True,batch_dimension=batch_along),
                        batch=True,
                        )
                    error = val_batch-rec
                    elementwise_norm = nla.norm(val_batch,axis=0)
                    error_norm = nla.norm(error,axis=0)
                    for _ in range(len(error.shape)-2):
                        elementwise_norm = nla.norm(elementwise_norm,axis=0)
                        error_norm = nla.norm(error_norm,axis=0)
                    val_errors.extend((error_norm/elementwise_norm).tolist())

                # Projection error for test videos
                for video_file in test_videos:
                    video = cv2.VideoCapture(video_file)
                    frames = []
                    while True:
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[...,None]
                        frames.append(frame)
                    test_batch = np.concatenate(frames,axis=-1)
                    test_batch = test_batch.reshape(resize_reshape, order=ORD)
                    rec = dataset.reconstruct(
                        dataset.project(test_batch,batch=True,batch_dimension=batch_along),
                        batch=True,
                        )
                    error = test_batch-rec
                    elementwise_norm = nla.norm(test_batch,axis=0)
                    error_norm = nla.norm(error,axis=0)
                    for _ in range(len(error.shape)-2):
                        elementwise_norm = nla.norm(elementwise_norm,axis=0)
                        error_norm = nla.norm(error_norm,axis=0)
                    test_errors.extend((error_norm/elementwise_norm).tolist())
                
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
            
            print(f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compression_ratio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}")
            batch_index += 1
                
        





    
    return

if __name__ == "__main__":
    args = get_args()


    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass
    
    if args.resize is None:
        print('Resize is not provided, using default resize: ', DEFAULT_RESIZE)
        args.resize = DEFAULT_RESIZE

    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for MineRL Basalt data.")
        args.reshaping = [2,4,4,4,8,8,2]
    
    assert np.prod(args.resize) == np.prod(args.reshaping), "Reshaping and resizing do not match."
    # args.reshaping.extend([sum(args.states)])
    # print("Reshaping used: ", args.reshaping)


    print(args)
    overall_tic = time.time()
    main(args)
    overall_time = time.time() - overall_tic
    print(f'Time for entire process (s): {round(overall_time,3)}')