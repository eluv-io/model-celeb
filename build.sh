#!/bin/bash
# buildscripts' `make build` always runs ./build.sh: dispatch to the container named by
# IMAGE_NAME (exported by Makefile.vector / Makefile.tagger).

set -e

cd "$(dirname "$(realpath "$0")")"

case "$IMAGE_NAME" in
    celeb-vector)        exec model-celeb-vector/build.sh ;;
    celeb-vector-tagger) exec model-celeb-vector-tagger/build.sh ;;
    *) echo "unknown IMAGE_NAME '$IMAGE_NAME': use make -f Makefile.vector or make -f Makefile.tagger" >&2; exit 1 ;;
esac
