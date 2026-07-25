#!/bin/bash

set -e

git submodule update --init --recursive

# This container reuses model-celeb's `celeb` package, so it must build from the parent
# model-celeb/ dir (the Containerfile COPYs `celeb` and `models` from there).
SCRIPT_PATH="$(dirname "$(realpath "$0")")"
PARENT_PATH="$(dirname "$SCRIPT_PATH")"

MODEL_PATH=$(yq -r .storage.model_path $SCRIPT_PATH/config.yml)
# weights land in the parent's models/ dir — the build context root
# exclude the celebrity pool: don't need image_features/ or cast lookup
rsync --progress --update --times --recursive --links --delete \
  --exclude 'image_features' --exclude 'ca_lookup.json' \
  $MODEL_PATH/ $PARENT_PATH/models/

cd "$PARENT_PATH"
exec buildscripts/build_container.bash -t "celeb-vector:${IMAGE_TAG:-latest}" -f model-celeb-vector/Containerfile .
