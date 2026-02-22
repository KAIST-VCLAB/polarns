#!/bin/bash
echo "================================================================"
HOST_CODE_DIR="/mnt/d/Project/polar-denoise/FBANet"

NAME="fbanet"
DEVICE="device=0"
PORT="16008"

docker run -ti --ipc=host \
--name $NAME \
--gpus $DEVICE \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v $HOST_CODE_DIR:/root/code \
-v /mnt/d/Data:/root/data \
-p $PORT:$PORT \
--privileged \
fbanet:latest
