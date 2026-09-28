# model-celeb-vector-tagger

Names the faces that [model-celeb-vector](../model-celeb-vector) wrote to the vectorstore: it reads those embeddings back, matches each one against a ground truth pool, and emits celebrity tags. Swapping or growing the pool only reruns this step, a matmul per face, instead of reprocessing the video.

The code is `src.tagger` in the repo's shared [`src`](../src) package. This folder holds only the container's entrypoint, config and build.

## How it runs

It is a **content aligned processor** (`type: processor`, `content_aligned: True` in the tagger config). The tagger does not download media for it. Instead, on stdin it sends one JSON file per time chunk of the content (default 600 s), such as `0000000000_0000600000.json`, containing `{"start_time": 0, "end_time": 600000}` in ms.

For each chunk, the container:

1. Fetches every vector of the content on `vector_track` that starts in `[start_time, end_time)`, with one `POST /indexes/{ELV_INDEX_QID}/search/binary` request.
2. Matches each vector to its most similar pool face with an exact fp32 search on CPU (numpy).
3. Keeps faces whose score is at least `thres` and whose name passes the cast filter.
4. Emits the tags, all with `source_media` set to the chunk file and times on the content timeline:
   - **one tag per face**, with `additional_info: {"box", "frame_idx", "confidence"}`
   - **one merged tag per run of adjacent sampled frames that show the same name**. Frames count as adjacent if they are at most 1.5 sample intervals apart. A run ends one sample interval after its last frame.

`frame_idx` goes in `additional_info` because the tagger drops `frame_info` for processor models, which have no fps.

## Environment

| Variable | Required | |
|---|---|---|
| `ELV_INDEX_QID` | yes | vectorstore index holding model-celeb-vector's embeddings |
| `ELV_CONTENT` | yes | content being tagged; the tagger sets it for every container |
| `ELV_TOKEN` | yes | auth for the vectorstore; also used to download a pool that isn't bundled |
| `ELV_VECTORSTORE_URL` | no | defaults to `vectorstore.url` in [config.yml](config.yml) |

## Runtime parameters (`--params` JSON string)

- `thres` (float, default -1): minimum similarity to the best matching pool face. -1 uses 0.55 for IBC and 0.4 for any other pool.
- `ground_truth` (default `IBC`): a pool bundled under `models/image_features`, or a content id to download with `ELV_TOKEN`.
- `content_id`, `restrict_list`: restrict names to a cast. The precedence is: the pool's `ca_lookup.json` entry for `content_id`, else `restrict_list`, else the pool's `restrict.txt`.
- `vector_track` (default `face_vectors`): the vectorstore track model-celeb-vector wrote to. This is its model name in the tagger config.
- `sample_interval_ms` (default: inferred): the spacing of model-celeb-vector's sampled frames, used for merging. By default it is inferred as the smallest gap between face timestamps in a chunk.

## Tagger config

```yaml
  celeb_from_vectors:
    image: "localhost/celeb-vector-tagger:latest"
    type: "processor"
    description: Celebrity Identification from Face Vectors
    category: "Frame Level Detection"
    resources: {
      "cpu_juice": 10
    }
    content_aligned: True
    track_outputs: ["celebrity_detection"]
```

The tagger currently only passes `ELV_TOKEN` and `ELV_CONTENT` to containers (see `src/tag_containers/containers.py`), so it also has to pass `ELV_INDEX_QID`.

## Build and test

From the repo root:

```
make -f Makefile.tagger build    # syncs the pools from storage.gt_path, builds celeb-vector-tagger:latest
make -f Makefile.tagger test     # pytest tests/test_tagger.py
```

The tests use a tiny synthetic pool and a fake vectorstore, so they need no pool or network.

This image does no detection or embedding and needs no GPU: matching is numpy on CPU. With the IBC pool (1M faces), matching 1000 faces takes about 4 s using 8 BLAS threads. The Containerfile caps BLAS at 8 threads with `OPENBLAS_NUM_THREADS`; more threads burn far more CPU for no speedup. Loading the pool takes about 2 GB of RAM.
