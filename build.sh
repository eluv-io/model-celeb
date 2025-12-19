#!/bin/bash

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
##rm -rf $SCRIPT_PATH/models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH/ $SCRIPT_PATH/models/
buildscripts/build_container.bash -t "celeb:${IMAGE_TAG:-latest}" -f Containerfile .
