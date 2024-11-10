#!/bin/bash

docker build -f ./docker/Dockerfile --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) --build-arg USERNAME=$USER -t nerf:latest .
