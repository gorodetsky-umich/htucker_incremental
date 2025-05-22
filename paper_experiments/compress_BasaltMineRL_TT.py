#!.env/bin/python -u
import datetime
import glob
import os
import random
import sys
import time

import cv2
import numpy as np

import wandb

sys.path.insert(0, "/home/doruk/TT-ICE/")
import DaMAT as dmt
import numpy.linalg as nla
from compress_BasaltMineRL_HT import get_args, initialize_wandb

MAX_SEED = 2**32 - 1
CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
TRAIN_RATIO = 0.997
VAL_RATIO = 0.002
TEST_RATIO = 0.002
ORD = "F"
DEFAULT_RESIZE = [128, 128]
HEUR_2_USE = ["skip", "occupancy"]
OCCUPANCY = 1


def compress_BasaltMineRL_TT(args):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    tags = ["BasaltMineRL", "TT-ICE", "LH", f"{args.resize[0]}x{args.resize[1]}"]
    if args.wandb:
        initialize_wandb(args, timestamp=timestamp, tags=tags, method="TT")

    random.seed(args.seed_idx)  # Fix the seed to a random number
    np.random.seed(args.seed_idx)  # Fix the seed to a random number

    video_files = glob.glob(os.path.join(args.data_location, "*.mp4"))

    total_time = 0
    frame_counter = 0
    video_counter = 0
    total_frame_counter = 0
    resize_reshape = args.reshaping + [3, -1]  # 3 for RGB channels

    train_videos = random.sample(video_files, int(TRAIN_RATIO * len(video_files)))
    val_videos = random.sample(
        list(set(video_files) - set(train_videos)), int(VAL_RATIO * len(video_files))
    )
    test_videos = list(set(video_files) - set(train_videos) - set(val_videos))
    print(resize_reshape)
    print(len(train_videos), len(val_videos), len(test_videos))
    frames = []
    train_video = cv2.VideoCapture(train_videos[0])
    for _ in range(args.batch_size):
        success_train, frame = train_video.read()
        if not success_train:
            break
        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[..., None]
        frames.append(frame)
    train_batch = np.concatenate(frames, axis=-1)
    print(train_batch.shape)
    train_batch = train_batch.reshape(resize_reshape, order=ORD)
    print(train_batch.shape)
    batch_norm = nla.norm(train_batch)
    error_before_update = 0

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
    previous_ranks = dataset.ttRanks.copy()
    batch_index = 0
    rec = dataset.reconstruct(dataset.ttCores[-1][:, -args.batch_size :, 0]).squeeze(0)
    frame_counter += args.batch_size
    total_frame_counter += args.batch_size
    error_after_update = nla.norm(rec - train_batch) / batch_norm
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
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[..., None]
            frames.append(frame)
        try:
            val_batch = np.concatenate(frames, axis=-1)
        except ValueError:
            continue
        val_batch = val_batch.reshape(resize_reshape, order=ORD)
        rec = dataset.reconstruct(
            dataset.projectTensor(
                val_batch,
            ),
        ).squeeze(0)
        error = val_batch - rec
        elementwise_norm = nla.norm(val_batch, axis=0)
        error_norm = nla.norm(error, axis=0)
        for _ in range(len(error.shape) - 2):
            elementwise_norm = nla.norm(elementwise_norm, axis=0)
            error_norm = nla.norm(error_norm, axis=0)
        val_errors.extend(error_norm / (elementwise_norm).tolist())

    # Projection error for test videos
    for video_file in test_videos:
        video = cv2.VideoCapture(video_file)
        frames = []
        while True:
            success, frame = video.read()
            if not success:
                break
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[..., None]
            frames.append(frame)
        try:
            test_batch = np.concatenate(frames, axis=-1)
        except ValueError:
            continue
        test_batch = test_batch.reshape(resize_reshape, order=ORD)
        rec = dataset.reconstruct(
            dataset.projectTensor(
                test_batch,
            ),
        ).squeeze(0)
        error = test_batch - rec
        elementwise_norm = nla.norm(test_batch, axis=0)
        error_norm = nla.norm(error, axis=0)
        for _ in range(len(error.shape) - 2):
            elementwise_norm = nla.norm(elementwise_norm, axis=0)
            error_norm = nla.norm(error_norm, axis=0)
        test_errors.extend((error_norm / elementwise_norm).tolist())
    # return
    ranks = dataset.ttRanks[1:-1].copy()
    logging_dict = {
        "compression_ratio": dataset.compressionRatio,
        "error_before_update": error_before_update,
        "error_after_update": error_after_update,
        "image_count": total_frame_counter,
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
        f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
    )
    batch_index += 1
    while success_train:
        frames = []
        for _ in range(args.batch_size):
            success_train, frame = train_video.read()
            if not success_train:
                break
            frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[..., None]
            frames.append(frame)
        try:
            train_batch = np.concatenate(frames, axis=-1)
        except Exception:
            break
        train_batch = train_batch.reshape(resize_reshape, order=ORD)
        batch_norm = nla.norm(train_batch)
        projection = dataset.reconstruct(
            dataset.projectTensor(
                train_batch,
            ),
        ).squeeze(0)
        error_before_update = nla.norm(projection - train_batch) / batch_norm
        tic = time.time()
        dataset.ttICEstar(
            train_batch,
            heuristicsToUse=HEUR_2_USE,
            epsilon=args.epsilon,
            occupancyThreshold=OCCUPANCY,
        )
        batch_time = time.time() - tic
        update_flag = dataset.ttRanks != previous_ranks
        total_time += batch_time
        rec = dataset.reconstruct(
            dataset.projectTensor(
                train_batch,
            ),
        ).squeeze(0)
        error_after_update = nla.norm(rec - train_batch) / batch_norm
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
                    frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[
                        ..., None
                    ]
                    frames.append(frame)
                try:
                    val_batch = np.concatenate(frames, axis=-1)
                except ValueError:
                    continue
                val_batch = val_batch.reshape(resize_reshape, order=ORD)
                rec = dataset.reconstruct(
                    dataset.projectTensor(
                        val_batch,
                    ),
                ).squeeze(0)
                error = val_batch - rec
                elementwise_norm = nla.norm(val_batch, axis=0)
                error_norm = nla.norm(error, axis=0)
                for _ in range(len(error.shape) - 2):
                    elementwise_norm = nla.norm(elementwise_norm, axis=0)
                    error_norm = nla.norm(error_norm, axis=0)
                val_errors.extend((error_norm / elementwise_norm).tolist())

            # Projection error for test videos
            for video_file in test_videos:
                video = cv2.VideoCapture(video_file)
                frames = []
                while True:
                    success, frame = video.read()
                    if not success:
                        break
                    frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[
                        ..., None
                    ]
                    frames.append(frame)
                try:
                    test_batch = np.concatenate(frames, axis=-1)
                except ValueError:
                    continue
                test_batch = test_batch.reshape(resize_reshape, order=ORD)
                rec = dataset.reconstruct(
                    dataset.projectTensor(
                        test_batch,
                    ),
                ).squeeze(0)
                error = test_batch - rec
                elementwise_norm = nla.norm(test_batch, axis=0)
                error_norm = nla.norm(error, axis=0)
                for _ in range(len(error.shape) - 2):
                    elementwise_norm = nla.norm(elementwise_norm, axis=0)
                    error_norm = nla.norm(error_norm, axis=0)
                test_errors.extend((error_norm / elementwise_norm).tolist())

            ranks = dataset.ttRanks[1:-1].copy()
        logging_dict = {
            "compression_ratio": dataset.compressionRatio,
            "error_before_update": error_before_update,
            "error_after_update": error_after_update,
            "image_count": total_frame_counter,
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
            f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )
        previous_ranks = dataset.ttRanks.copy()

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
                frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[..., None]
                frames.append(frame)
            try:
                train_batch = np.concatenate(frames, axis=-1)
            except ValueError:
                break
            train_batch = train_batch.reshape(resize_reshape, order=ORD)
            batch_norm = nla.norm(train_batch)
            projection = dataset.reconstruct(
                dataset.projectTensor(
                    train_batch,
                ),
            ).squeeze(0)
            error_before_update = nla.norm(projection - train_batch) / batch_norm
            tic = time.time()
            dataset.ttICEstar(
                train_batch,
                heuristicsToUse=HEUR_2_USE,
                epsilon=args.epsilon,
                occupancyThreshold=OCCUPANCY,
            )
            batch_time = time.time() - tic
            total_time += batch_time
            update_flag = dataset.ttRanks != previous_ranks
            rec = dataset.reconstruct(
                dataset.projectTensor(
                    train_batch,
                ),
            ).squeeze(0)
            error_after_update = nla.norm(rec - train_batch) / batch_norm
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
                        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[
                            ..., None
                        ]
                        frames.append(frame)
                    try:
                        val_batch = np.concatenate(frames, axis=-1)
                    except ValueError:
                        continue
                    val_batch = val_batch.reshape(resize_reshape, order=ORD)
                    rec = dataset.reconstruct(
                        dataset.projectTensor(
                            val_batch,
                        ),
                    ).squeeze(0)
                    error = val_batch - rec
                    elementwise_norm = nla.norm(val_batch, axis=0)
                    error_norm = nla.norm(error, axis=0)
                    for _ in range(len(error.shape) - 2):
                        elementwise_norm = nla.norm(elementwise_norm, axis=0)
                        error_norm = nla.norm(error_norm, axis=0)
                    val_errors.extend((error_norm / elementwise_norm).tolist())

                # Projection error for test videos
                for video_file in test_videos:
                    video = cv2.VideoCapture(video_file)
                    frames = []
                    while True:
                        success, frame = video.read()
                        if not success:
                            break
                        frame = cv2.resize(frame, args.resize, interpolation=cv2.INTER_LINEAR)[
                            ..., None
                        ]
                        frames.append(frame)
                    try:
                        test_batch = np.concatenate(frames, axis=-1)
                    except ValueError:
                        continue
                    test_batch = test_batch.reshape(resize_reshape, order=ORD)
                    rec = dataset.reconstruct(
                        dataset.projectTensor(
                            test_batch,
                        ),
                    ).squeeze(0)
                    error = test_batch - rec
                    elementwise_norm = nla.norm(test_batch, axis=0)
                    error_norm = nla.norm(error, axis=0)
                    for _ in range(len(error.shape) - 2):
                        elementwise_norm = nla.norm(elementwise_norm, axis=0)
                        error_norm = nla.norm(error_norm, axis=0)
                    test_errors.extend((error_norm / elementwise_norm).tolist())

                ranks = dataset.ttRanks[1:-1].copy()

            logging_dict = {
                "compression_ratio": dataset.compressionRatio,
                "error_before_update": error_before_update,
                "error_after_update": error_after_update,
                "image_count": total_frame_counter,
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
                f"{batch_index:6d} {total_frame_counter:6d} {video_counter:3d} {frame_counter:4d} {round(batch_time, 5):08.5f} {round(total_time, 5):09.5f} {batch_norm:14.5f} {round(error_before_update, 5):0.5f} {round(error_after_update, 5):0.5f} {round(dataset.compressionRatio, 5):09.5f} {round(np.mean(val_errors),5):0.5f} {round(np.mean(test_errors),5):0.5f} {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
            )
            batch_index += 1
            previous_ranks = dataset.ttRanks.copy()


if __name__ == "__main__":
    args = get_args()

    if args.seed_idx is None:
        rng = np.random.Generator(np.random.PCG64DXSM())
        args.seed_idx = int(rng.integers(MAX_SEED))
    else:
        pass

    if args.resize is None:
        print("Resize is not provided, using default resize: ", DEFAULT_RESIZE)
        args.resize = DEFAULT_RESIZE

    if args.reshaping == []:
        print("Reshaping is not provided, using baseline reshaping for MineRL Basalt data.")
        args.reshaping = [2, 4, 4, 4, 8, 8, 2]

    assert np.prod(args.resize) == np.prod(args.reshaping), "Reshaping and resizing do not match."
    # args.reshaping.extend([sum(args.states)])
    # print("Reshaping used: ", args.reshaping)

    print(args)
    overall_tic = time.time()
    compress_BasaltMineRL_TT(args)
    overall_time = time.time() - overall_tic
    print(f"Time for entire process (s): {round(overall_time,3)}")
