#!/bin/bash

# Start TensorBoard in the background
tensorboard --logdir lightning_logs --host 0.0.0.0 --port 6006 &

# Forward all input arguments to the training script
python main.py "$@"
