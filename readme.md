
# Neural Radiance Field Research Repo
A repository containing my research into current nerf methods

## Setup
Clone repo

    git submodule update --init --recursive

## Generate initial sparse reconstruction

Build colmap image

    /colmap/build_docker.sh

Create a sparse reconstriction of a set of images

    /colmap/docker_run.sh /path/to/data

## Run using docker

Update add or update a config file in the config folder. Run `./docker/build.sh` to create the docker image and then run `./docker/run.sh <path/to dataset> <path/to/config.yaml>` to start a training session or  `./docker/run.sh <path/to dataset> <path/to/config.yaml> <path/to/ckpt> --visualize` to export
a mesh from the nerf.

E.g.

`
./docker/run.sh /database config/nerf_config.yaml
`

## Run using vscode
Download vscode and install Dev containers

Update devcontainer.json to map the folder containing database to the /database folder inside the container

    -v=/path/to/database/:/database:rw

Update add or update a config file in the config folder and then launch the nerf task for training and the nerf vis task for a 3d mesh export
