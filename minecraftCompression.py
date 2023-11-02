import os
import cv2
import glob
import time
import random

import numpy as np
import htucker as ht

cwd = os.getcwd()

directories = {
    "video" : "/Users/doruk/data/",
    "core"  : "/media/doruk/Database/minecraftCores/",
    "save"  : "/.",
    "metric": cwd+"/experiments/minecraft",
}

videoFiles = glob.glob(directories["video"]+"*.mp4")
print(len(videoFiles))
ord = "F"
epsilon = 0.3
initialize = 1
increment = 1
newShape = [6,6,10,10,8,8,3]
newShape = [3,4,5,6,10,8,8,3]
newShape = [18,20,32,20,3]
resizeShape = [128,128]
resizeReshape = [4,8,8,8,8,3]
resizeReshape = [2,2,2,4,4,2,4,2,2,4,3]
resizeReshape = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,3]

totTime = 0
frameCtr = 1
videoCtr = 0 
totFrameCtr = 1 

video = cv2.VideoCapture(videoFiles[0])
success , image = video.read()
# image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
# print(type(image),image.dtype,image.shape)
# image = image.reshape(resizeReshape, order=ord)[...,None]
image = image.reshape(newShape, order=ord)[...,None]
batch_along = len(image.shape)-1
print(image.shape)

frames = ht.HTucker()
frames.initialize(
    image,
    batch = True,
    batch_dimension = batch_along,
)
dim_tree = ht.createDimensionTree(
    frames.original_shape[:batch_along]+frames.original_shape[batch_along+1:],
    numSplits = 2,
    minSplitSize =1,
)
dim_tree.get_items_from_level()
frames.rtol = epsilon
tic = time.time()
frames.compress_leaf2root_batch(
    image,
    dimension_tree = dim_tree,
    batch_dimension = batch_along,
)

toc = time.time()-tic
totTime += toc
# print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
# print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
print(frames.root.core.shape)

while success:
    success , image = video.read()
    try:
        # image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
        # image = image.reshape(resizeReshape, order=ord)[...,None]
        image = image.reshape(newShape, order=ord)[...,None]
        tic = time.time()
        frames.incremental_update_batch(
            image,
            batch_dimension = batch_along,
            append = True,
        )
        toc = time.time()-tic
        totTime += toc
        updCtr += updFlag*1
        # print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
        # print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
        print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
        frameCtr += 1
        totFrameCtr += 1
    except cv2.error:
        pass
print(videoCtr,frameCtr,totFrameCtr)
print(frames.root.core.shape)
videoCtr +=1

for videoFile in videoFiles[1:]:
    video = cv2.VideoCapture(videoFiles[0])
    success , image = video.read()
    # image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
    # image = image.reshape(resizeReshape, order=ord)[...,None]
    # print(image.shape,type(image))
    image = image.reshape(newShape, order=ord)[...,None]
    tic = time.time()
    updFlag = frames.incremental_update_batch(
        image,
        batch_dimension = batch_along,
        append = True,
    )
    toc = time.time()-tic
    totTime += toc
    updCtr += updFlag*1
    # print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
    # print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
    print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
    while success:
        success , image = video.read()
        try:
            # image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
            # image = image.reshape(resizeReshape, order=ord)[...,None]
            image = image.reshape(newShape, order=ord)[...,None]
            tic = time.time()
            frames.incremental_update_batch(
                image,
                batch_dimension = batch_along,
                append = True,
            )
            toc = time.time()-tic
            totTime += toc
            # print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
            # print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
            print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
            frameCtr += 1
            totFrameCtr += 1
        except cv2.error:
            pass
    print(videoCtr,frameCtr,totFrameCtr)
    print(frames.root.core.shape)
    videoCtr +=1
