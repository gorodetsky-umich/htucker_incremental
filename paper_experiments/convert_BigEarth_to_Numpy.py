#!.env/bin/python
import argparse
import glob
import os
from functools import partial, reduce
from multiprocessing import Pool

import cv2
import numpy as np
import rasterio
from compress_BigEarthHT import read_image
from tqdm import tqdm

CWD = os.getcwd()
PATH_SEP = os.path.sep
HOME = os.path.expanduser("~")
ORD = "F"
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def main():
    parser = argparse.ArgumentParser(
        description="This script reads the BigEarthNet image patches\
            and writes them into NumPy files"
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

    if args.data_location is None:
        if os.path.exists(
            reduce(
                os.path.join, [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0"]
            )
        ):
            print("Data location is not provided, using default location")
            data_loc = reduce(
                os.path.join, [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0"]
            )
        else:
            raise IsADirectoryError("Please provide the data location")
    else:
        data_loc = args.data_location

    if args.save_location is None:
        print("Save location is not provided, using default location")
        save_loc = reduce(
            os.path.join, [PATH_SEP] + HOME.split(os.path.sep) + ["data", "BigEarthNet-v1.0_numpy"]
        )
    else:
        save_loc = args.save_location
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    directories = {
        "data": data_loc + PATH_SEP,
        "save": save_loc + PATH_SEP,
    }
    print(directories["save"])

    image_folders = glob.glob(directories["data"] + "*")
    print("Number of total images: ", len(image_folders))
    image_size = cv2.imread(
        image_folders[0] + PATH_SEP + os.listdir(image_folders[0])[0], cv2.IMREAD_UNCHANGED
    ).shape

    for band in BAND_NAMES:
        f = rasterio.open(
            image_folders[0] + PATH_SEP + image_folders[0].split(PATH_SEP)[-1] + f"_{band}.tif"
        )
        if np.prod(f.read(1).shape) > np.prod(image_size):
            image_size = f.read(1).shape
        f.close()
    print("Maximum image size: ", image_size)
    for image_folder in tqdm(image_folders):
        image_name = os.path.split(image_folder)[-1]
        if os.path.exists(directories["save"] + image_name + ".npy"):
            continue
        # print("Reading image: ", image_name)
        with Pool() as pool:
            image = pool.map(
                partial(
                    read_image,
                    image_path=image_folder,
                    resizing=image_size,
                    interpolation_method=cv2.INTER_AREA,
                ),
                BAND_NAMES,
            )
        image = np.array(image).transpose(1, 2, 0)
        # print("Image shape: ", image.shape)
        np.save(directories["save"] + image_name, image)


if __name__ == "__main__":
    main()
