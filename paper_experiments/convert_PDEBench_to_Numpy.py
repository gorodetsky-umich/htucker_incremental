#!.env/bin/python
import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
ORD = "F"
STATES = ["Vx", "Vy", "Vz", "density", "pressure"]


def main():
    parser = argparse.ArgumentParser(
        description="This script reads the PDEBench simulations from hdf5 files and \
            writes them into NumPy files"
    )
    parser.add_argument(
        "-d", "--data_location", dest="data_location", help="path to data", default=None
    )
    parser.add_argument(
        "-s",
        "--save_location",
        dest="save_location",
        help="path to save the NumPy files",
        default=None,
    )

    args = parser.parse_args()
    print(args)

    file_name = args.data_location.split(PATH_SEP)[-1]
    initial_condition = file_name.split("_")[2]
    M = file_name.split("M")[-1].split("_")[0]
    print(initial_condition)
    print(M)
    print(os.path.split(args.data_location)[:-1][0])

    # return
    if args.save_location is None:
        print(
            "Save location is not provided, writing under the same directory \
              as the input data"
        )
        save_loc = os.path.split(args.data_location)[:-1][0]
    else:
        save_loc = args.save_location

    save_loc = save_loc + PATH_SEP + f"PDEBench{PATH_SEP}{initial_condition}{PATH_SEP}{M}{PATH_SEP}"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    with h5py.File(args.data_location, "r") as f:
        num_simulations = f[STATES[0]].shape[0]
        grid_shape = list(f[STATES[0]].shape[-3:])
        num_timesteps = f[STATES[0]].shape[1]
        num_states = len(STATES)
        print(num_simulations, grid_shape, num_timesteps, num_states)
        sim = np.zeros(grid_shape + [num_states])
        print(sim.shape)
        for sim_idx in tqdm(range(num_simulations)):
            for timestep in range(num_timesteps):
                for state_idx, state in enumerate(STATES):
                    sim[..., state_idx] = f[state][sim_idx][timestep]
                    np.save(
                        save_loc + f"sim{sim_idx:03d}_ts{timestep:02d}",
                        sim,
                    )


if __name__ == "__main__":
    main()
