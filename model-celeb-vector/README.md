# model-celeb-vector

A face **vector embedding** tagger. Detects faces (MTCNN) and emits one InsightFace r100
embedding per face as a vector tag.

## Relationship to model-celeb

This is a subdirectory of `model-celeb/` and **reuses its `celeb` package unchanged**
(`FaceModel`, the InsightFace r100 embedder). It standardizes image and video embeddings to use InsightFace r100 for one comparable index in the vector DB. It persists the raw embeddings to a vector DB so adding to the ground truth pool becomes fast matrix-vecotr multiplication against the stored vectors rather than a full reprocess (`model-celeb` detects, embeds, and matches against a celebrity pool in a single pass). Identity matching and clustering move downstream to query time.

## Output

Each detected face becomes a `Tag` with:
- `vector`: L2-normalized InsightFace embedding (cosine == dot product)
- `box`: normalized face box `{x1, y1, x2, y2}`
- `tag`: empty (identity is resolved downstream)
- `additional_info`: `{"model": "insightface-r100-ii", "dim": 512}` for provenance/validation

The following are also added automatically by `common-ml`'s `AVModel.from_frame_model`:
- `frame_info.frame_idx`: source video frame index the face was detected in
- `start_time` / `end_time`: that frame's timestamp in ms (equal, since vectors are per-frame)
- `source_media`: the input file path

`common-ml`'s `AVModel.from_frame_model` does not combine adjacently for per-frame vector tags.

## Runtime parameters (`--config` JSON string)

- `min_box_size` (float, default 0): drop faces whose normalized box area is below this
- `det_confidence` (float, default 0.96): MTCNN confidence gate
- `fps` (default 1.0) and `allow_single_frame` (default False): the standard frame-model params in `common-ml`

## Build

Because it reuses the parent's `celeb` package, the build context is the **parent
`model-celeb/` dir**. From `model-celeb-vector/` dir:

```
chmod +x build.sh
./build.sh        # rsyncs weights into ../models and builds from the parent context
```

or manually, from `model-celeb/`:

```
podman build -f model-celeb-vector/Containerfile -t celeb-vector .
```

## Test

From `model-celeb-vector/`:

```
chmod +x test.sh
./test.sh                          # container smoke test against ../test-files
pytest tests/                      # unit test (needs weights + deps available locally)
```

## Downstream: matching stored vectors against the pool

This container only stores embeddings; turning them into celebrity names now happens downstream
against a pool. This step lives inside `model-celeb` ([`_tag_frames`](../celeb/model.py)) and is now cheap and re-runnable, because the expensive detect+embed work is already done and saved here.

Inputs — the same ground-truth pool files `model-celeb` fetches:
- `feats_*.npy` — pool of celebrity face vectors, `(m, 512)` float16, L2-normalized
- `gt_*.npy` — pool id per row `(m,)`
- `id2name_*.json` — id → display name
- optionally a per-content cast list, to restrict candidates

### Matching

For stored face vectors `F` `(n, 512)` and pool `P` `(m, 512)` (both unit-normalized),
the whole match is now one matrix multiply:

```python
P  = np.load("feats_with_ibc.npy").astype(np.float32)     # (m, 512)
gt = np.load("gt_with_ibc.npy", allow_pickle=True)        # (m,)
id2name = json.load(open("id2name_with_ibc.json"))

F = np.stack(face_vectors).astype(np.float32)             # (n, 512) from stored tags
S = F @ P.T                                               # (n, m) cosine (both are unit vectors)
top    = S.argmax(1)
scores = S[np.arange(len(F)), top]
names  = [id2name.get(str(gt[t]), "") for t in top]
```

(Can filter to vectors whose `additional_info.model` matches the pool's model, so you only
compare within one embedding space (a top-k ANN query per face).)

### Thresholding

Keep a match only if its cosine `>= threshold`; below it, the face is "unknown".
`model-celeb`'s defaults: **0.55** for the IBC pool, **0.4** otherwise
([model.py](../celeb/model.py) `_add_params`). Can now tune per pool.

### Pooling

(Distinct from the celebrity pool `P`.)

One person appearing across many frames → many stored vectors.
Mean-pooling the vectors belonging to the same person before matching (or matching each and voting) gives a more stable identity than any single frame. This mirrors `model-celeb`'s per-cluster mean score.

### Clustering

Optional, query-time — the same idea as `model-celeb`'s `clustering()` (connected
components over a similarity graph), but run over stored vectors instead of in the tagger:
- **Within an asset**: build a face-to-face cosine similarity graph, threshold the edges,
  take connected components → groups of the same person. Assign a name per group by
  majority vote and propagate it to faces that individually fell below threshold (recovers
  hard frames — the original reason `model-celeb` clusters).
- **Across the corpus**: cluster all stored vectors to discover unique individuals, dedupe,
  or build per-person galleries — now feasible because the vector face embeddings are persisted.

### Adding to the pool

When a new celebrity is added: embed their reference face(s) with the **same InsightFace
r100 model**, append the row(s) to `feats`/`gt`/`id2name`, and re-run the matmul (or ANN
query) over the already-stored face vectors. Source media is no longer re-processed (re-decoded or re-embedded) (the payoff of vectorizing `model-celeb`).
