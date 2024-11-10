
# Neural Radiance Field Research Repo
A repository containing my research into current nerf methods

## Setup
Clone repo

    git submodule update --init --recursive

Build colmap image

    /colmap/build_docker.sh

## Sparse reconstruction
Create a sparse reconstriction of a set of images

    /colmap/docker_run.sh /path/to/data

## Run using docker

Run `./docker/build.sh` to create the nerf image and then run `./docker/run.sh <path/to dataset> <path/to/config.yaml>` to start a training session or  `./docker/run.sh <path/to dataset> <path/to/config.yaml> <path/to/ckpt> --visualize` to export
a mesh from the nerf.

E.g.

`
./docker/run.sh /database config/nerf_config.yaml
`

## Run using vscode
Download vscode and install Dev containers

Update devcontainer.json to map the folder containing database to the /database folder inside the container

    -v=/path/to/database/:/database:rw

Update the dataset path in the config file to point to the reconstruction

    dataset_path: /database/path/to/data/dense

Launch nerf task for training and nerf vis task for an 3d mesh export
