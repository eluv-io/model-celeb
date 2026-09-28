# model-celeb-vector

Detects faces (MTCNN) and emits one InsightFace r100 embedding per face as a vector tag. The tagger writes these vectors to the vectorstore, and [model-celeb-vector-tagger](../model-celeb-vector-tagger) later names them against a ground truth pool.

The code is `src.vector` in the repo's shared [`src`](../src) package. This folder holds only the container's entrypoint, config and build.

## Output

Each detected face becomes a `Tag` with:
- `vector`: the L2-normalized InsightFace embedding (512 floats; cosine == dot product)
- `box` and `additional_info.box`: the normalized face box `{x1, y1, x2, y2}`
- `tag`: empty, since names are resolved by model-celeb-vector-tagger

common-ml's `AVModel.from_frame_model` adds these:
- `frame_info.frame_idx`: the source video frame the face was detected in
- `start_time` / `end_time`: that frame's timestamp in ms (equal, since vectors are per frame)
- `source_media`: the input file path

## Runtime parameters (`--params` JSON string)

- `min_box_size` (float, default 0.1): drop faces whose normalized box area is below this
- `det_confidence` (float, default 0.96): MTCNN confidence gate
- `fps` (default 1.0) and `allow_single_frame`: common-ml's standard frame-model params

## Requirements

- **Python 3.8–3.9 only.** `mxnet-cu101==1.9.1`, `torch==1.9.0` and `numpy<1.20` have no wheels beyond cp39, and mxnet needs a CUDA 10.1 userspace. The container pins 3.8.
- **GPU:** runs on GPUs torch 1.9 supports (up to Turing, e.g. RTX 6000 or T4). Newer ones, such as the L40S, fall back to CPU.
- **Weights:** the InsightFace r100 weights are expected at `models/models/model-r100-ii/`. Fetch them with [`pull-models`](../pull-models); `build.sh` syncs them from `storage.model_path`.

## Build and test

From the repo root:

```
make -f Makefile.vector build    # celeb-vector:latest
make -f Makefile.vector test     # runs the container over test-files/
pytest tests/test_vector.py      # in-process test (needs the weights and a pip install ".[vector]" env)
```
