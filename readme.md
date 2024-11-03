
# Neural Radiance Field Research Repo
A repository containing my research into current nerf methods

# Installation Instructions
The installation and usage are described in this section 

## Setup
Clone repo

    git submodule update --init --recursive

Build colmap image

    /colmap/build_docker.sh

## Sparse reconstruction
Create a sparse reconstriction of a set of images

    /colmap/docker_run.sh /path/to/data

where the images are in a subfolder in data called images

## Run and visualize nerf
Download vscode and install Dev containers

Update devcontainer.json to map the folder containing database to the /database folder inside the container

    -v=/path/to/database/:/database:rw

Update the dataset path in the config file to point to the reconstruction

    dataset_path: /database/path/to/data/dense

Launch nerf task for training and nerf vis task for an 3d mesh export
