#!/bin/bash

### This script contains commands to run the experiments for the Incremental Hierarchical Tucker Decomposition paper. Note that for datasets without an explicit train-test split, we repeated the experiments for 5 different random seeds. The results reported in the paper are the average of these 5 runs. The seeds used for each dataset are listed below. ###

### PDEBench ###
# seed 1: 1905241905
# seed 2: 3393210734
# seed 3: 2308261220
# seed 4: 2122106210
# seed 5: 2125608406

## Epsilon = 0.10 ##
## HT-RISE ##
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.10 -s <seed index> -N none -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.10 -s <seed index> -N maxabs -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.10 -s <seed index> -N zscore -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.10 -s <seed index> -N unitvector -r 8 8 8 8 8 8

## TT-ICE ##
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.1 -s <seed index> -N none
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.1 -s <seed index> -N maxabs
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.1 -s <seed index> -N zscore
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.1 -s <seed index> -N unitvector

## Epsilon = 0.05 ##
## HT-RISE ##
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N none -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N maxabs -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N zscore -r 8 8 8 8 8 8
# ./compress_PDEBench_HT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N unitvector -r 8 8 8 8 8 8

## TT-ICE ##
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N none
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N maxabs
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N zscore
# ./compress_PDEBench_TT.py -n -c -d <path to PDEBench data> -t turb -b 1 -e 0.05 -s <seed index> -N unitvector


### Self Oscillating Gels ###
# seed 1: 1905241905

## Epsilon = 0.10 ##
## HT-RISE ##
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n none
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n maxabs
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n zscore
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n unitvector

## TT-ICE ##
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n none
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n maxabs
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n zscore
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.10 -s 1905241905 -n unitvector


## Epsilon = 0.01 ##
## HT-RISE ##
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n none
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n maxabs
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n zscore
# ./compress_CatGel_HT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n unitvector

## TT-ICE ##
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n none
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n maxabs
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n zscore
# ./compress_CatGel_TT.py -d <path to CatGel data> -t 1 -e 0.01 -s 1905241905 -n unitvector


### Basalt Mine RL ###
# seed 1: 1543767237
# seed 2: 2066790544
# seed 3: 2077296768
# seed 4: 2171494882
# seed 5: 3932108494

## Epsilon = 0.10 ##
# ./compress_BasaltMineRL_HT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.1
# ./compress_BasaltMineRL_TT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.1

## Epsilon = 0.20 ##
# ./compress_BasaltMineRL_HT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.2
# ./compress_BasaltMineRL_TT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.2

## Epsilon = 0.30 ##
# ./compress_BasaltMineRL_HT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.3
# ./compress_BasaltMineRL_TT.py -d <path to MineRL data> -r 2 4 4 4 8 8 2 -b 20 -s <seed index> -e 0.3


### BigEarthNet ###
# seed 1: 1604208352
# seed 2: 2674378496
# seed 3: 2752113014
# seed 4: 3190267801
# seed 5: 3305108268

## Epsilon = 0.05 ##
# ./compress_BigEarthHT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.05
# ./compress_BigEarthTT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.05

## Epsilon = 0.10 ##
# ./compress_BigEarthHT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.1
# ./compress_BigEarthTT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.1

## Epsilon = 0.15 ##
# ./compress_BigEarthHT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.15
# ./compress_BigEarthTT.py -n -r 12 10 12 10 10 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.15



######################################################################################################################

###### HT VS BHT COMPARISONS ######

### BigEarthNet ###
# seed 1: 1905241905
# seed 2: 3393210734
# seed 3: 2308261220
# seed 4: 2122106210
# seed 5: 2125608406

# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.05 -m ht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.10 -m ht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.15 -m ht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.30 -m ht

# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.05 -m bht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.10 -m bht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.15 -m bht
# ./compare_HT_BHT_BigEarth.py -n -r 12 10 12 10 12 -b 100 -d <path to BigEarthNet data> -s <seed index> -e 0.30 -m bht


### PDEBench ###

## Epsilon = 0.10 ##
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N maxabs --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N none --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N unitvector --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N zscore --method ht

# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N maxabs --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N none --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N unitvector --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.1 -s 2125608406 -N zscore --method bht

## Epsilon = 0.05 ##
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N maxabs --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N none --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N unitvector --method ht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N zscore --method ht

# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N maxabs --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N none --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N unitvector --method bht
# ./compare_HT_BHT_PDEBench.py -n -c -d /nfs/turbo/coe-goroda/hierarchicalTucker/PDEBench -t turb -b 1 -w -e 0.05 -s 2125608406 -N zscore --method bht
