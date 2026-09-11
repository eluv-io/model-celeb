#!/bin/bash

set -e

git submodule update --init --recursive

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
##rm -rf $SCRIPT_PATH/models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH/ $SCRIPT_PATH/models/
buildscripts/build_container.bash -t "celeb-vector:${IMAGE_TAG:-latest}" -f model-celeb-vector/Containerfile .
