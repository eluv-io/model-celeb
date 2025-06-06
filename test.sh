#!/bin/bash

[ "$ELV_MODEL_TEST_GPU_TO_USE" != "" ] || ELV_MODEL_TEST_GPU_TO_USE=0

set -x

rm -rf test_output/
mkdir test_output

mkdir -p .cache-ro

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache-ro:/root/.cache:ro --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE celeb test/*.mp4

podman run --rm --volume=$(pwd)/test:/elv/test:ro --volume=$(pwd)/test_output:/elv/tags --volume=$(pwd)/.cache-ro:/root/.cache:ro --network host --device nvidia.com/gpu=$ELV_MODEL_TEST_GPU_TO_USE celeb test/*.png

ex=$?

cd test_output

find

exit $ex

