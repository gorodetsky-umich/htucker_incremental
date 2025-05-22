# Experiments of Incremental Hierarchical Tucker Decomposition

This folder contains the necessary scripts to run the experiments of the manuscript [Incremental Hierarchical Tucker Decomposition](https://arxiv.org/abs/2412.16544). Just like in the manuscript, the experiments can be grouped under the following 2 categories based on the datasets used:

- Scientific datasets
- Image-based datasets

## Required Python Packages

To run the experiment scripts in this folder, the following Python packages are required:

- numpy
- h5py
- wandb (optional, for experiment tracking)
- tqdm
- opencv-python (cv2)
- rasterio
- pillow (PIL)
- htucker (custom package, should be installed from this repository)
- argparse
- datetime
- random
- glob
- time
- os
- sys
- functools
- multiprocessing

You can install the packages that are not included in the standard python library using pip:

```bash
pip install -r requirements-paper-experiments.txt
```

The `htucker` package will also be installed after running the command above.

## Collecting the necessary data for experiments

As of May 2025, all datasets considered in this work are open-source and available using the following links

- [Self-oscillating Gel Simulation Snapshot Dataset](https://deepblue.lib.umich.edu/data/concern/data_sets/d791sh19r)
- [PDEBench 3D Navier-Stokes Simulations with Turbulent ICs](https://darus.uni-stuttgart.de/file.xhtml?fileId=164694&version=8.0)
- [BigEarthNet Satellite Image archive](https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1)

The videos for the BasaltMineRL competition dataset can be downloaded directly using
```bash
bash ./download_minerl_data.sh <your download directory>
```
The list of videos included in the analyses is included in [`minecraft_video_list.txt`](./minecraft_video_list.txt). The script will automatically read and start downloading.

## Preparing the downloaded datasets

To accelerate the execution of the experiment scripts, I heavily recommend you to preprocess the PDEBench and BigEarthNet datasets.

```bash
./convert_BigEarth_to_numpy.py -d <BigEarthNet data location> -s <save location>
```
This script will convert the downloaded Sentinel 2 satellite images to NumPy arrays using `rasterio` package.

```bash
./convet_PDEBench_to_Numpy.py -d <BigEarthNet data location> -s <save location>
```
This script will convert the turbulent PDEBench Navier-Stokes snapshots from hdf5 format to NumPy arrays using `h5py` package.

## Running the experiments

An overarching umbrella script is provided in [`run_experiments.sh`](./run_experiments.sh). Please check out the comments to learn how to run the corresponding experiments.
