#!/bin/bash
# Builds celeb-vector from the repo root (run via `make -f Makefile.vector build`).

set -e

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
cd "$(dirname "$SCRIPT_PATH")"

git submodule update --init --recursive

# the InsightFace weights are the only models this container needs
MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
mkdir -p models/models
rsync --progress --update --times --recursive --links --delete $MODEL_PATH/models/ models/models/
buildscripts/build_container.bash -t "celeb-vector:${IMAGE_TAG:-latest}" -f model-celeb-vector/Containerfile .
