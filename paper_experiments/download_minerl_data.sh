#!/bin/bash
# This script is adapted from https://github.com/minerllabs/basalt-benchmark/blob/main/scripts/download_demonstration_data.sh to download the demonstration data for MineRL Basalt experiments.
# Usage: bash download_demonstration_data.sh <basalt_data_dir> [max_data_size_in_mb_per_task]
# Example: bash download_demonstration_data.sh data_directory 1000
# This will create following folder structure:
# <base_data_dir>
#     MineRLBasaltFindCave-v0
#

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash download_demonstration_data.sh <basalt_data_dir> [max_data_size_in_mb_per_task]"
    echo "       max_data_size_in_mb_per_task is optional, specifying it will limit the size of each task's data."
    echo "       Example: bash download_demonstration_data.sh data_directory 1000"
    echo "       This will download total of 4 x 1000 MB of data, 1000 MB per task."
    exit 1
fi

# Create arguments for wget
if [ $# -eq 2 ]; then
    WGET_ARGS="--quota=${2}m"
else
    WGET_ARGS=""
fi

# Filelists are next to this script, so get the absolute path
FILELIST_DIR=$(dirname $(readlink -f $0))

DATA_DIR=$1
# mkdir -p $DATA_DIR

# FindCave data
mkdir -p $DATA_DIR/MineRLBasaltFindCave-v0/
wget -nc -i $FILELIST_DIR/minecraft_video_list.txt -P $DATA_DIR/MineRLBasaltFindCave-v0 $WGET_ARGS || true
