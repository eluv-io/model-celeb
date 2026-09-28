#!/bin/bash
# Builds celeb-vector-tagger from the repo root (run via `make -f Makefile.tagger build`).

set -e

SCRIPT_PATH="$(dirname "$(realpath "$0")")"
cd "$(dirname "$SCRIPT_PATH")"

git submodule update --init --recursive

# the ground truth pools are the only models this container needs
GT_PATH=$(yq -r .storage.gt_path $SCRIPT_PATH/config.yml)
mkdir -p models/image_features
rsync --progress --update --times --recursive --links --delete $GT_PATH/ models/image_features/
buildscripts/build_container.bash -t "celeb-vector-tagger:${IMAGE_TAG:-latest}" -f model-celeb-vector-tagger/Containerfile .
