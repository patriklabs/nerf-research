
# Neural Radiance Field Research Repo
A repository containing my research into current nerf methods

# Installation Instructions

Clone repo

    git submodule update --init --recursive

Build colmap image

    /colmap/build_docker.sh

Create a sparse reconstriction of a set of images

    /colmap/docker_run.sh /path/to/image-folder

Download vscode and install Dev containers

Update devcontainer.json to map the folder containing database to the /database folder inside the container

    -v=/path/to/database/:/database:rw

Update the dataset path launch.json or tasks.json

    --dataset_path=/database/path/to/image-folder

to point to the reconstruction

Launch nerf for training and nerf vis for an 3d mesh export
