#!/bin/bash

DATASET_PATH=/data

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.camera_model "PINHOLE" \
   --ImageReader.single_camera 1

colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse