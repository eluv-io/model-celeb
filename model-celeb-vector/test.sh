#!/bin/bash
#
# Local smoke test for the celeb-vector container.
#
# The tagger reads input file paths on STDIN
# (run_default reads stdin, not argv), --output-path is required, and tunables like fps go
# in as a --params JSON string.
#
#   ./test.sh          # fps=1 (default)
#   ./test.sh 5        # fps=5

set -uo pipefail

FPS="${1:-1}"
: "${ELV_MODEL_TEST_GPU_TO_USE:=3}"
IMAGE_NAME="${IMAGE_NAME:-celeb-vector}"

cd "$(dirname "$0")"

set -x

rm -rf test_output/
mkdir -p test_output

# test-files live in the parent model-celeb/ repo; bind-mount them read-only at /elv/test
# and feed newline-separated container paths on stdin.
INPUT=$(find ../test-files -maxdepth 1 -type f | sed 's|^.*/test-files/|/elv/test/|' | sort)

echo "$INPUT" | podman run --rm -i \
    --volume="$(pwd)/../test-files:/elv/test:ro" \
    --volume="$(pwd)/test_output:/elv/tags:U" \
    --network host \
    --device "nvidia.com/gpu=${ELV_MODEL_TEST_GPU_TO_USE}" \
    "${IMAGE_NAME}" \
    --output-path /elv/tags/out.jsonl \
    --params "{\"fps\": ${FPS}}"

ex=$?

set +x
echo "=== test_output/out.jsonl ==="
find test_output -type f

exit $ex
