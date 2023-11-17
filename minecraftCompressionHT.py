import os
import sys
import cv2
import glob
import time
import random

import numpy as np
import htucker as ht

cwd = os.getcwd()

directories = {
    "video"     : "/nfs/turbo/coe-goroda/minecraftData/",
    "core"      : "/nfs/turbo/coe-goroda/minecraftCores/",
    "save"      : "/nfs/turbo/coe-goroda/minecraftCores/",
    "metrics"   : "./",
    "image"     : cwd+"/frames/",
    "videoOrder": cwd+"/",
}

if os.path.exists("minecraftVideoOrder.txt"):
    videoFiles = []
    with open(directories["videoOrder"]+"minecraftVideoOrder.txt", 'r') as f:
        for line in f:
            videoFiles.append(directories["video"]+line[:-1])

else:
    videoFiles = glob.glob(directories["video"]+"*.mp4")
    # print(videoFiles)
    with open(directories["videoOrder"]+"minecraftVideoOrder.txt", 'w') as f:
        for line in videoFiles:
            f.writelines(line.split("/")[-1]+"\n")
# print(videoFiles)
ord = "F"
epsilon = 0.3
epsilon = float(sys.argv[1])
initialize = 1
increment = 1

resizeShape = [128,128]
resizeReshape = [2,4,4,4,8,8,2,3]

saveName = f"minecraft_128x128x3_eps"+"".join(f"{epsilon:0.2f}_".split("."))+"_".join(map(str,resizeReshape))
metricsFile = "minecraft_128x128x3_eps"+"".join(f"{epsilon:0.2f}_".split("."))+"_".join(map(str,resizeReshape))+".txt"


errors = []
lines2print = []

totTime = 0
frameCtr = 0
videoCtr = 0 
totFrameCtr = 0

video = cv2.VideoCapture(videoFiles[0])
success , image = video.read()
image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
image = image.reshape(resizeReshape, order=ord)[...,None]
batch_along = len(image.shape)-1

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


lines2print.append(f"{videoCtr}")
lines2print.append(f"{frameCtr}")
lines2print.append(f"{totFrameCtr}")
lines2print.append(f"{toc}")
lines2print.append(f"{totTime}")
lines2print.append(f"{frames.compression_ratio}")
print(lines2print)
print(" ".join(map(str,lines2print)))



frameCtr += 1
totFrameCtr += 1
# print(frames.reconstruct(frames.project(
#                 image,
#                 batch = True,
#                 batch_dimension = batch_along
#                 )).shape,image.shape)
imageNorm = np.linalg.norm(image)
ranks =[list(frames.root.shape)]
for core in frames.transfer_nodes:
    ranks.append(list(core.shape))
for leaf in frames.leaves:
    ranks.append(list(leaf.shape))
# print(" ".join(map(str,ranks)))
lines2print.append(
    np.linalg.norm(frames.reconstruct(
            frames.project(
                image,
                batch = True,
                batch_dimension = batch_along
                ))-image)/imageNorm
    )
lines2print.append(imageNorm)
# print(lines2print)
# print(" ".join(map(str,lines2print)))
# print(" ".join(map(str,lines2print))+" "+" ".join(map(str,ranks))+"\n")
with open(directories["metrics"]+metricsFile, 'a') as f:
    f.writelines(" ".join(map(str,lines2print))+" "+" ".join(map(str,ranks))+"\n")
# print(" ".join(map(str,lines2print))+" "+f"{*ranks}")
# quit()
updCtr = 0
while success:
    success , image = video.read()
    lines2print = []
    try:
        image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(directories["image"]+f"original_ds/{frameCtr}.jpg",image)
        image = image.reshape(resizeReshape, order=ord)[...,None]
        # image = image.reshape(newShape, order=ord)[...,None]
        tic = time.time()
        updFlag = frames.incremental_update_batch(
            image,
            batch_dimension = batch_along,
            append = False,
        )
        toc = time.time()-tic
        # image_rec = frames.reconstruct(frames.root.core[...,-1]).reshape(resizeShape+[-1],order=ord).astype(np.uint8)
        # cv2.imwrite(directories["image"]+f"eps030/{frameCtr}.jpg",image_rec)
        totTime += toc
        updCtr += updFlag*1
        # print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
        # print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
        print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
        imageNorm = np.linalg.norm(image)
        # errors.append(
        #     [
        #         np.linalg.norm(frames.reconstruct(
        #             frames.project(
        #                 image,
        #                 batch = True,
        #                 batch_dimension = batch_along
        #                 ))-image)/imageNorm,
        #         imageNorm
        #     ]
        #     )
        lines2print.append(f"{videoCtr}")
        lines2print.append(f"{frameCtr}")
        lines2print.append(f"{totFrameCtr}")
        lines2print.append(f"{toc}")
        lines2print.append(f"{totTime}")
        lines2print.append(f"{frames.compression_ratio}")
        lines2print.append(
            np.linalg.norm(frames.reconstruct(
                    frames.project(
                        image,
                        batch = True,
                        batch_dimension = batch_along
                        ))-image)/imageNorm
            )
        lines2print.append(imageNorm)
        
        frameCtr += 1
        totFrameCtr += 1
        
        if updFlag:
            # An update is performed
            ranks =[list(frames.root.shape)]
            for core in frames.transfer_nodes:
                ranks.append(list(core.shape))
            for leaf in frames.leaves:
                ranks.append(list(leaf.shape))
        else:
            # No update is performed
            pass
        with open(directories["metrics"]+metricsFile, 'a') as f:
            f.writelines(" ".join(map(str,lines2print))+" "+" ".join(map(str,ranks))+"\n")
    except cv2.error:
        pass
    except AttributeError:
    #     # print(2)
        pass
frames.save(
    saveName,
    fileType = "hto",
    directory = "./"
    # directory = directories["core"]
)

for videoFile in videoFiles[1:]:
    lines2print = []
    videoCtr += 1
    frameCtr = 0 
    video = cv2.VideoCapture(videoFile)
    success , image = video.read()
    image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
    image = image.reshape(resizeReshape, order=ord)[...,None]
    tic = time.time()
    updFlag = frames.incremental_update_batch(
        image,
        batch_dimension = batch_along,
        append = False,
    )
    toc = time.time()-tic
    totTime += toc
    updCtr += updFlag*1

    imageNorm = np.linalg.norm(image)
    lines2print.append(f"{videoCtr}")
    lines2print.append(f"{frameCtr}")
    lines2print.append(f"{totFrameCtr}")
    lines2print.append(f"{toc}")
    lines2print.append(f"{totTime}")
    lines2print.append(f"{frames.compression_ratio}")
    lines2print.append(
        np.linalg.norm(frames.reconstruct(
                frames.project(
                    image,
                    batch = True,
                    batch_dimension = batch_along
                    ))-image)/imageNorm
        )
    lines2print.append(imageNorm)
    frameCtr += 1
    totFrameCtr += 1
    if updFlag:
        # An update is performed
        ranks =[list(frames.root.shape)]
        for core in frames.transfer_nodes:
            ranks.append(list(core.shape))
        for leaf in frames.leaves:
            ranks.append(list(leaf.shape))
    else:
        # No update is performed
        pass
    with open(directories["metrics"]+metricsFile, 'a') as f:
        f.writelines(" ".join(map(str,lines2print))+" "+" ".join(map(str,ranks))+"\n")
    while success:
        success , image = video.read()
        lines2print = []
        try:
            image = cv2.resize(image,resizeShape,interpolation=cv2.INTER_LINEAR)
            # cv2.imwrite(directories["image"]+f"original_ds/{frameCtr}.jpg",image)
            image = image.reshape(resizeReshape, order=ord)[...,None]
            # image = image.reshape(newShape, order=ord)[...,None]
            tic = time.time()
            updFlag = frames.incremental_update_batch(
                image,
                batch_dimension = batch_along,
                append = False,
            )
            toc = time.time()-tic
            # image_rec = frames.reconstruct(frames.root.core[...,-1]).reshape(resizeShape+[-1],order=ord).astype(np.uint8)
            # cv2.imwrite(directories["image"]+f"eps030/{frameCtr}.jpg",image_rec)
            totTime += toc
            updCtr += updFlag*1
            # print(f"Compressed in: {round(toc , 3)}. Total time: {round(totTime , 3)}")
            # print(f"Compression ratio: {round(frames.compression_ratio , 4)}")
            print(f"{videoCtr} {frameCtr} {round(toc , 3)} {round(totTime , 3)} {round(frames.compression_ratio, 4)}")
            imageNorm = np.linalg.norm(image)
            # errors.append(
            #     [
            #         np.linalg.norm(frames.reconstruct(
            #             frames.project(
            #                 image,
            #                 batch = True,
            #                 batch_dimension = batch_along
            #                 ))-image)/imageNorm,
            #         imageNorm
            #     ]
            #     )
            lines2print.append(f"{videoCtr}")
            lines2print.append(f"{frameCtr}")
            lines2print.append(f"{totFrameCtr}")
            lines2print.append(f"{toc}")
            lines2print.append(f"{totTime}")
            lines2print.append(f"{frames.compression_ratio}")
            lines2print.append(
                np.linalg.norm(frames.reconstruct(
                        frames.project(
                            image,
                            batch = True,
                            batch_dimension = batch_along
                            ))-image)/imageNorm
                )
            lines2print.append(imageNorm)
            
            frameCtr += 1
            totFrameCtr += 1
            
            if updFlag:
                # An update is performed
                ranks =[list(frames.root.shape)]
                for core in frames.transfer_nodes:
                    ranks.append(list(core.shape))
                for leaf in frames.leaves:
                    ranks.append(list(leaf.shape))
            else:
                # No update is performed
                pass
            with open(directories["metrics"]+metricsFile, 'a') as f:
                f.writelines(" ".join(map(str,lines2print))+" "+" ".join(map(str,ranks))+"\n")
        except cv2.error:
            pass
        except AttributeError:
        #     # print(2)
            pass
    frames.save(
        saveName,
        fileType = "hto",
        directory = "./"
        # directory = directories["core"]
    )


# print(errors)




