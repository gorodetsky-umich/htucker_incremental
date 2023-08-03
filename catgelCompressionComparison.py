import os
import glob
import time
import random

import numpy as np
import DaMAT as dmt
import htucker as ht

cwd = os.getcwd()

ord = "F" 
method = "ttsvd"
epsilon = 0.01
initialize = 1
increment = 1
motion = 1

tt = True
htucker = True

assert (tt^htucker), "Either one of htucker or tt should be True!"

checkTestErr = True
checkTrainErr = True
checkPointwiseErr = True
writeMode = True


trainDM = "3rd"
testDM = "5th"

dataLocation = {
    "train" : f"/home/doruk/bayesianOED/catgelData/{trainDM}/{motion}/",
    "test"  : f"/home/doruk/bayesianOED/catgelData/{testDM}/{motion}/",
}
testSamples = 15000
trainSamples = 6400

data = np.load(dataLocation["train"]+f"catgelData_{0}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
batch_along=len(data.shape)-1
dataNorm = np.linalg.norm(data)

metricsFileName = {
    "ht" : f"ht_E"+"".join(format(epsilon,'.2f').split("."))+f"M{motion}"+"S"+"_".join(map(str,data.shape))+".txt",
    "tt" : f"tt_E"+"".join(format(epsilon,'.2f').split("."))+f"M{motion}"+"S"+"_".join(map(str,data.shape))+".txt"
}

print(data.shape)
lines2print = []

if htucker:
    ht_trainError=[]
    tens = ht.HTucker()
    tens.initialize(
        data,
        batch=True,
        batch_dimension=batch_along
    )

    dim_tree=ht.createDimensionTree(
        tens.original_shape[:batch_along]+tens.original_shape[batch_along+1:],
        numSplits=2,
        minSplitSize=1
    )
    dim_tree.get_items_from_level()
    tens.rtol=epsilon
    tic = time.time()
    tens.compress_leaf2root_batch(
        data,
        dimension_tree = dim_tree,
        batch_dimension = batch_along
    )
    HT_toc = time.time()-tic 
    HT_time = HT_toc
    data_rec = tens.reconstruct(tens.root.core[:,:,-1])
    data_relErr = np.linalg.norm(data_rec-data)/dataNorm
    ht_trainError.append(
        (dataNorm, data_relErr)
    )
    ht_pwError_pre = "-"
    ht_pwError_post = data_relErr
    ht_trainError_overall = np.sqrt(np.square(np.prod(ht_trainError,axis=-1)).sum())/np.sqrt(np.square(ht_trainError).sum(0)[0])
    ht_trainError_average = np.mean(ht_trainError,axis=0)[-1]
    if writeMode:
        lines2print.append(f"{data.shape[-1]}") # Number of samples in batch
        lines2print.append(f"{dataNorm}") # Norm of the batch
        lines2print.append(f"{HT_toc}") # Time to update
        lines2print.append(f"{ht_pwError_pre}") # Projection error before update
        lines2print.append(f"{ht_pwError_post}") # Projection error after update
        lines2print.append(f"{ht_trainError_overall}") # Overall training error
        lines2print.append(f"{ht_trainError_average}") # Average training error
    else:
        print(f"HT Compression Ratio {round(tens.compression_ratio,4)}")
        print(f"HT Step Time {round(HT_toc,4)}")

if tt:
    tt_trainError=[]
    dataSet = dmt.ttObject(
        data,
        epsilon=epsilon,
        keepData=False,
        samplesAlongLastDimension=True,
        method=method
    )

    tic = time.time()
    dataSet.ttDecomp()
    TT_toc = time.time()-tic 
    TT_time = TT_toc
    last_ranks = dataSet.ttRanks.copy()
    tt_pwError_pre = "-"
    tt_pwError_post = dataSet.computeRelError(data)
    tt_trainError.append(
        (np.linalg.norm(data), tt_pwError_post)
    )
    tt_trainError_overall = np.sqrt(np.square(np.prod(tt_trainError,axis=-1)).sum())/np.sqrt(np.square(tt_trainError).sum(0)[0])
    tt_trainError_average = np.mean(tt_trainError,axis=0)[-1]
    if writeMode:
        lines2print.append(f"{data.shape[-1]}") # Number of samples in batch
        lines2print.append(f"{dataNorm}") # Norm of the batch
        lines2print.append(f"{TT_toc}") # Time to update
        lines2print.append(f"{tt_pwError_pre}") # Projection error before update
        lines2print.append(f"{tt_pwError_post}") # Projection error after update
        lines2print.append(f"{tt_trainError_overall}") # Overall training error
        lines2print.append(f"{tt_trainError_average}") # Average training error
    else:
        print(f"TT Compression Ratio {round(dataSet.compressionRatio,4)}")
        print(f"TT Step Time {round(TT_toc,4)}")

if checkTestErr:
    if htucker:
        ht_testError = []
        for testIdx in range(testSamples):
            data = np.load(dataLocation["test"]+f"catgelData_{testIdx}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
            dataNorm = np.linalg.norm(data)
            ht_testError.append(
                (dataNorm, np.linalg.norm(tens.reconstruct(tens.project(data))-data)/dataNorm)
            )
        ht_testError_overall = np.sqrt(np.square(np.prod(ht_testError,axis=-1)).sum())/np.sqrt(np.square(ht_testError).sum(0)[0])
        ht_testError_average = np.mean(ht_testError,axis=0)[-1]
    if tt:
        tt_testError = []
        for testIdx in range(testSamples):
            data = np.load(dataLocation["test"]+f"catgelData_{testIdx}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
            dataNorm = np.linalg.norm(data)
            tt_testError.append(
                (dataNorm, dataSet.computeRelError(data))
            )
        tt_testError_overall = np.sqrt(np.square(np.prod(tt_testError,axis=-1)).sum())/np.sqrt(np.square(tt_testError).sum(0)[0])
        tt_testError_average = np.mean(tt_testError,axis=0)[-1]
else:
    if htucker:
        ht_testError_overall = "-"
        ht_testError_average = "-"
    if tt:
        tt_testError_overall = "-"
        tt_testError_average = "-"

if writeMode:
    if htucker:
        lines2print.append(f"{ht_testError_overall}") # Overall test error
        lines2print.append(f"{ht_testError_average}") # Average test error
    if tt:
        lines2print.append(f"{tt_testError_overall}") # Overall test error
        lines2print.append(f"{tt_testError_average}") # Average test error

for simIdx in range(1,trainSamples):
    # print()
    print(simIdx)
    data = np.load(dataLocation["train"]+f"catgelData_{simIdx}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
    dataNorm = np.linalg.norm(data)
    if tt:
        if checkPointwiseErr:
            tt_pwError_pre = dataSet.computeRelError(data)
        else:
            tt_pwError_pre = "-"
        tic = time.time()
        dataSet.ttICEstar(
            data,
            epsilon=epsilon,
            heuristicsToUse=["skip"]
            )
        TT_toc = time.time()-tic 
        TT_time += TT_toc
        if checkPointwiseErr:
            tt_pwError_post = dataSet.computeRelError(data)
        else:
            tt_pwError_post = "-"
        tt_trainError.append(
            (dataNorm, tt_pwError_post)
        )
        tt_trainError_overall = np.sqrt(np.square(np.prod(tt_trainError,axis=-1)).sum())/np.sqrt(np.square(tt_trainError).sum(0)[0])
        tt_trainError_average = np.mean(tt_trainError,axis=0)[-1]
        if writeMode:
            lines2print.append(f"{data.shape[-1]}") # Number of samples in batch
            lines2print.append(f"{dataNorm}") # Norm of the batch
            lines2print.append(f"{TT_toc}") # Time to update
            lines2print.append(f"{tt_pwError_pre}") # Projection error before update
            lines2print.append(f"{tt_pwError_post}") # Projection error after update
            lines2print.append(f"{tt_trainError_overall}") # Overall training error
            lines2print.append(f"{tt_trainError_average}") # Average training error

    # print(f"TT Compression Ratio {round(dataSet.compressionRatio,4)}")
    # print(f"TT Step Time {round(TT_toc,4)}")
    # print(f"TT Total Time {round(TT_time,4)}")


    if htucker:
        if checkPointwiseErr:
            ht_pwError_pre = np.linalg.norm(tens.reconstruct(tens.project(data))-data)/dataNorm
        else:
            ht_pwError_pre = "-"
        tic = time.time()
        ht_updateFlag = tens.incremental_update_batch(
            data,
            batch_dimension=batch_along,
            append=True
        )
        HT_toc = time.time()-tic 
        HT_time += HT_toc
        data_rec = tens.reconstruct(tens.root.core[:,:,-1])
        data_relErr = np.linalg.norm(data_rec-data)/dataNorm
        ht_trainError.append(
            (dataNorm, data_relErr)
        )
        if checkPointwiseErr:
            ht_pwError_post = data_relErr
        else:
            ht_pwError_post = "-"
        ht_trainError_overall = np.sqrt(np.square(np.prod(ht_trainError,axis=-1)).sum())/np.sqrt(np.square(ht_trainError).sum(0)[0])
        ht_trainError_average = np.mean(ht_trainError,axis=0)[-1]
        if writeMode:
            lines2print.append(f"{data.shape[-1]}") # Number of samples in batch
            lines2print.append(f"{dataNorm}") # Norm of the batch
            lines2print.append(f"{TT_toc}") # Time to update
            lines2print.append(f"{ht_pwError_pre}") # Projection error before update
            lines2print.append(f"{ht_pwError_post}") # Projection error after update
            lines2print.append(f"{ht_trainError_overall}") # Overall training error
            lines2print.append(f"{ht_trainError_average}") # Average training error
    if checkTestErr:
        if htucker:
            if ht_updateFlag:
                pass
            else:
                ht_testError = []
                for testIdx in range(testSamples):
                    data = np.load(dataLocation["test"]+f"catgelData_{testIdx}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
                    dataNorm = np.linalg.norm(data)
                    ht_testError.append(
                        (dataNorm, np.linalg.norm(tens.reconstruct(tens.project(data))-data)/dataNorm)
                    )
                ht_testError_overall = np.sqrt(np.square(np.prod(ht_testError,axis=-1)).sum())/np.sqrt(np.square(ht_testError).sum(0)[0])
                ht_testError_average = np.mean(ht_testError,axis=0)[-1]
        if tt:
            if last_ranks == dataSet.ttRanks:
                pass
            else:
                tt_testError = []
                for testIdx in range(testSamples):
                    data = np.load(dataLocation["test"]+f"catgelData_{testIdx}.npy").transpose(0,1,2,4,3)#.reshape(3367,10,3,-1).transpose(1,2,0,3)#[...,None]
                    dataNorm = np.linalg.norm(data)
                    tt_testError.append(
                        (dataNorm, dataSet.computeRelError(data))
                    )
                tt_testError_overall = np.sqrt(np.square(np.prod(tt_testError,axis=-1)).sum())/np.sqrt(np.square(tt_testError).sum(0)[0])
                tt_testError_average = np.mean(tt_testError,axis=0)[-1]
    else:
        if htucker:
            ht_testError_overall = "-"
            ht_testError_average = "-"
        if tt:
            tt_testError_overall = "-"
            tt_testError_average = "-"
    if writeMode:
        if htucker:
            lines2print.append(f"{ht_testError_overall}")
            lines2print.append(f"{ht_testError_average}")
        if tt:
            lines2print.append(f"{tt_testError_overall}")
            lines2print.append(f"{tt_testError_average}")

    last_ranks = dataSet.ttRanks.copy()
    # print(f"HT Compression Ratio {round(tens.compression_ratio,4)}")
    # print(f"HT Step Time {round(HT_toc,4)}")
    # print(f"HT Total Time {round(HT_time,4)}")
    
print(f"TT Compression Ratio {round(dataSet.compressionRatio,4)}")
print(f"TT Total Time {round(TT_time,4)}")
print(f"HT Compression Ratio {round(tens.compression_ratio,4)}")
print(f"HT Total Time {round(HT_time,4)}")



