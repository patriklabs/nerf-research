#!/bin/bash

# Usage message with optional validate flag and ckpt
usage="Usage: $0 <database_path> <config_path> [--visualize] [ckpt]"

# Check for minimum number of arguments
if [ "$#" -lt 2 ]; then
    echo "$usage"
    exit 1
fi

# Parse arguments
database_path="$1"
config_path="$2"
validate_flag=""
ckpt=""

# Check for optional --visualize flag and ckpt
for arg in "${@:3}"; do
    if [ "$arg" == "--visualize" ]; then
        validate_flag="--visualize"
    else
        ckpt="$arg"
    fi
done

# Check if Docker image exists
if ! docker image inspect nerf:latest >/dev/null 2>&1; then
    echo "Error: Docker image 'nerf:latest' not found. Please build or pull the image first."
    exit 1
fi

# Build Docker run command with optional validate flag and ckpt
docker_cmd="docker run -p 6006:6006 -u \"$(id -u):$(id -g)\" --rm -it \
    -v \"$database_path\":/database:rw \
    -v \"$(pwd)\":/workspace:rw \
    --shm-size 16G \
    --gpus=all \
    nerf:latest --config \"$config_path\" $validate_flag"

# Add ckpt if provided
if [ -n "$ckpt" ]; then
    docker_cmd+=" --ckpt \"$ckpt\""
fi

# Run the Docker command
eval "$docker_cmd"
