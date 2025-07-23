#!/usr/bin/env bash
set -xeo pipefail

BUILD_ID=$1

DOCKER_IMAGE=mirrors.tencent.com/hunyuan_infer/text2text_infer:H-5.0.1-v7-hpc-ops
DATE=$(date +"%Y_%m_%d")

CONTAINER_NAME=hpc-builder-$BUILD_ID

sudo docker run --entrypoint bash --privileged --network host --ipc host -v /:/host  \
    --shm-size=1024g --ulimit memlock=-1 \
    --name $CONTAINER_NAME \
    "$DOCKER_IMAGE" -c "/host${CICD_DIR}/build.sh"
