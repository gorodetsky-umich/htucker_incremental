import os
import time
import random
import numpy as np
import htucker as ht

from PIL import Image
from functools import partial
from concurrent.futures import ThreadPoolExecutor

## Load batch of images.

def parallelLoader(imageDir, runIdximgIdx):
    return np.array(
        Image.open(f"{imageDir}{runIdximgIdx[0]}/{runIdximgIdx[1]}.png"), dtype="int16"
    )

cwd = os.getcwd()
tensor_shape = [14,15,16,10,3]
tensor_shape = [7,6,5,8,5,4,3]
tensor_shape = [210,160,3]
ord = "F"
epsilon = 0.1
initialize = 1
increment = 1
trainRunCount = 10
runs2Compress = []
compressedRuns=[]
indexes = []
game = "MsPacman"

imgDir=cwd+f'/{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'

trainRunIndices = np.arange(trainRunCount).tolist()
for _ in range(initialize):
    run=trainRunIndices.pop(0)
    runs2Compress.append(run)
    compressedRuns.append(run)

numIms = np.zeros(initialize)
for idx,runIdx in enumerate(runs2Compress):
    os.chdir(imgDir+f'{runIdx}')
    numIms[idx]=int(len(os.listdir()))
    indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
numImgs=int(numIms.sum())
imageDimension=np.array(Image.open('0.png')).shape
with ThreadPoolExecutor() as tpe:
    images=list(tpe.map(partial(parallelLoader,imgDir),indexes))
images=np.array(images).transpose(1,2,3,0).reshape(tensor_shape+[-1],order=ord)#.transpose(0,4,3,1,2,5)
totalNorm=np.linalg.norm(images)

# images=[
#     np.array(Image.open("31.png")).reshape(tensor_shape+[1],order=ord),
#     np.array(Image.open("32.png")).reshape(tensor_shape+[1],order=ord),
#     # np.array(Image.open("322.png")).reshape(tensor_shape+[1],order=ord)
#         ]
# images = np.concatenate(images,axis=-1)
batch_dimension = len(images.shape)-1
batch_count = images.shape[batch_dimension]
rtol = np.linalg.norm(images)*epsilon



# amcik=np.array(Image.open("322.png")).reshape(tensor_shape+[1],order=ord)


print(images.shape,batch_dimension,batch_count)

tens=ht.HTucker()
tens.initialize(images,batch=True, batch_dimension=batch_dimension)

# dim_tree=ht.createDimensionTree(tens.original_shape,numSplits=2,minSplitSize=1)
dim_tree=ht.createDimensionTree(
    tens.original_shape[:batch_dimension]+tens.original_shape[batch_dimension+1:],numSplits=2,minSplitSize=1
)
dim_tree.get_items_from_level()
print([len(lv) for lv in dim_tree._level_items])
tens.rtol=epsilon
tic=time.time()
tens.compress_leaf2root_batch(images,dimension_tree=dim_tree,batch_dimension=batch_dimension)
print(round(time.time()-tic,4))
print(tens.compression_ratio)

# amcik_proj = tens.project(images,batch=True,batch_dimension=batch_dimension)
# amcik_reco = tens.reconstruct(amcik_proj)
# curIms.append(prevIms[-1])
while trainRunIndices:
    runs2Compress=[]
    for _ in range(increment):
        run=trainRunIndices.pop(0)
        runs2Compress.append(run)
        compressedRuns.append(run)
    imgDir=cwd+f'/{game}NoFrameskip-v4/{game}NoFrameskip-v4-recorded_images-'
    indexes=[]
    numIms=np.zeros(increment)
    for idx,runIdx in enumerate(runs2Compress):
        os.chdir(imgDir+f'{runIdx}')
        numIms[idx]=int(len(os.listdir()))
        indexes.extend(zip([runIdx]*len(os.listdir()),range(len(os.listdir()))))
    numImgs=int(numIms.sum())
    with ThreadPoolExecutor() as tpe:
        upd_images=list(tpe.map(partial(parallelLoader,imgDir),indexes))
    upd_images=np.array(upd_images).transpose(1,2,3,0).reshape(tensor_shape+[-1],order=ord)#.transpose(0,4,3,1,2,5)
    print(upd_images.shape,batch_dimension,batch_count)
    # prevIms.append(prevIms[-1]+images.shape[-1])
    imagesNorm=np.linalg.norm(np.linalg.norm(np.linalg.norm(upd_images,axis=0),axis=0),axis=0)
    # totalNorm=np.sqrt(totalNorm**2+np.linalg.norm(imagesNorm)**2)
    relErrorBeforeUpdate = np.linalg.norm(tens.reconstruct(tens.project(upd_images,batch=True, batch_dimension=batch_dimension))-upd_images)/np.linalg.norm(imagesNorm)
    print(relErrorBeforeUpdate)
    # relErrorBeforeUpdate=dataSet.computeRelError(images,useExact=True)
    # elementwiseRelErrorBeforeUpdate=dataSet.computeRelError(images,useExact=False)

    tic=time.time()
    tens.incremental_update_batch(upd_images,batch_dimension=batch_dimension,append=True)
    print(round(time.time()-tic,4))
    print(tens.compression_ratio)
    relErrorAfterUpdate = np.linalg.norm(tens.reconstruct(tens.project(upd_images,batch=True, batch_dimension=batch_dimension))-upd_images)/np.linalg.norm(imagesNorm)
    print(relErrorAfterUpdate)
2+2