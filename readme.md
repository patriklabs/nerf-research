# Neural Radiance Field Research Repository
This repository contains research and tools for working with Neural Radiance Fields (NeRF) using current methods.

## Setup
1. **Clone the repository** (including submodules):

   ```bash
   git clone --recursive https://github.com/patriklabs/nerf-research.git
   ```

## Generate Initial Sparse Reconstruction

1. **Build the COLMAP Docker image**:

   ```bash
   ./colmap/build_docker.sh
   ```

2. **Create a sparse reconstruction of a dataset**:
   
   Run COLMAP on your data to generate a sparse reconstruction by specifying the path to your dataset:

   ```bash
   ./colmap/docker_run.sh /path/to/data
   ```

## Running NeRF with Docker

1. **Prepare Configuration**:
   
   - Add or update a configuration file in the `config` folder to specify your training parameters.

2. **Build and Run the Docker Image**:
   
   - **Build the Docker image**:

     ```bash
     ./docker/build.sh
     ```

   - **Start a training session**:

     ```bash
     ./docker/run.sh /path/to/dataset /path/to/config.yaml
     ```

   - **Optional**: To export a 3D mesh, specify a checkpoint file and add the `--visualize` flag:

     ```bash
     ./docker/run.sh /path/to/dataset /path/to/config.yaml /path/to/ckpt --visualize
     ```

   **Example**:

   ```bash
   ./docker/run.sh /database config/nerf_config.yaml
   ```

## Running NeRF with Visual Studio Code (VSCode)

1. **Set Up VSCode and Dev Containers**:
   
   - Download and install [VSCode](https://code.visualstudio.com/) if not already installed.
   - Install the Dev Containers extension in VSCode.

2. **Configure DevContainer**:
   
   - Update `devcontainer.json` to map your local data directory to the `/database` folder inside the container:

     ```json
     "runArgs": [
		"-v=/path/to/database:/database:rw",
     
     ```

3. **Run Training and Visualization Tasks**:
   
   - Update or add a configuration file in the `config` folder.
   - Launch the **NeRF Training** task to start training.
   - To export a 3D mesh, use the **NeRF Visualization** task.
