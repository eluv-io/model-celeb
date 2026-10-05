import json

import numpy as np
import pytest

from src.tagger.config import RuntimeConfig
from src.tagger.model import CelebVectorTagger
from src.tagger.pool import CelebPool
from src.tagger.vectorstore import StoredVector, parse_binary_search

DIM = 8


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def pool_path(tmp_path):
    # two identities, one pool face each, on orthogonal axes
    np.save(tmp_path / "feats.npy", np.stack([_unit(0), _unit(1)]).astype(np.float16))
    np.save(tmp_path / "gt.npy", np.array(["1", "2"]))
    (tmp_path / "id2name.json").write_text(json.dumps({"1": "Alice", "2": "Bob"}))
    return str(tmp_path)


class FakeVectorstore:
    def __init__(self, hits, vectors):
        self.hits, self.vectors, self.calls = hits, vectors, []

    def get_vectors(self, content_qid, track, start_time_gte, start_time_lte):
        self.calls.append((content_qid, track, start_time_gte, start_time_lte))
        keep = [i for i, h in enumerate(self.hits) if start_time_gte <= h.start_time <= start_time_lte]
        return [self.hits[i] for i in keep], self.vectors[keep]


def _hit(t, frame_idx):
    return StoredVector(start_time=t, end_time=t, frame_idx=frame_idx,
                        additional_info={"box": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}})


def _run(pool_path, tmp_path, hits, vectors, **cfg):
    pool = CelebPool(pool_path)
    vs = FakeVectorstore(hits, np.stack(vectors))
    model = CelebVectorTagger(pool, vs, content_qid="iq__content", cfg=RuntimeConfig(**cfg))
    interval = tmp_path / "0000000000_0000600000.json"
    interval.write_text(json.dumps({"start_time": 0, "end_time": 600000}))
    return model.tag(str(interval)), vs


def test_tags_merge_threshold_and_info(pool_path, tmp_path):
    near_alice = _unit(0) * 0.9 + _unit(2) * 0.1  # cos ~0.99 to Alice
    weak_bob = _unit(1) * 0.3 + _unit(3) * 0.95   # cos ~0.3 to Bob: below threshold
    hits = [_hit(0, 0), _hit(1000, 24), _hit(2000, 48), _hit(5000, 120), _hit(1000, 24)]
    vectors = [near_alice, near_alice, near_alice, near_alice, weak_bob]
    tags, vs = _run(pool_path, tmp_path, hits, vectors, thres=0.5)

    # the interval is queried as [start, end) on the configured track for the content
    assert vs.calls == [("iq__content", "face_vectors", 0, 599999)]

    frame_tags = [t for t in tags if t.frame_info]
    merged = [t for t in tags if not t.frame_info]
    assert [t.tag for t in frame_tags] == ["Alice"] * 4
    assert frame_tags[1].frame_info.frame_idx == 24
    assert frame_tags[1].frame_info.box == {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}
    assert frame_tags[1].additional_info["confidence"] > 0.9
    # 0/1000/2000 are adjacent samples (1s apart), 5000 is a separate appearance;
    # each run ends one video frame (5000ms / 120 frames = 42ms) after its last sample
    assert [(t.tag, t.start_time, t.end_time) for t in merged] == [("Alice", 0, 2042), ("Alice", 5000, 5042)]
    assert all(t.additional_info is None for t in merged)
    assert all(t.source_media.endswith("0000000000_0000600000.json") for t in tags)


def test_single_frame_runs(pool_path, tmp_path):
    # two Alice faces on frame 24 count as one sample; 1000/2000 are a run, 5000 stands alone
    hits = [_hit(1000, 24), _hit(1000, 24), _hit(2000, 48), _hit(5000, 120)]
    vectors = [_unit(0)] * 4
    tags, _ = _run(pool_path, tmp_path, hits, vectors, thres=0.5)
    assert [(t.start_time, t.end_time) for t in tags if not t.frame_info] == [(1000, 2042), (5000, 5042)]

    tags, _ = _run(pool_path, tmp_path, hits, vectors, thres=0.5, allow_single_frame=False)
    assert len([t for t in tags if t.frame_info]) == 4
    assert [(t.start_time, t.end_time) for t in tags if not t.frame_info] == [(1000, 2042)]


def test_cast_pool_filters_names(pool_path, tmp_path):
    hits = [_hit(0, 0), _hit(1000, 24)]
    tags, _ = _run(pool_path, tmp_path, hits, [_unit(0), _unit(1)], thres=0.5, restrict_list=["Bob"])
    assert {t.tag for t in tags} == {"Bob"}


def test_default_threshold_follows_model_celeb(pool_path):
    pool = CelebPool(pool_path)
    assert CelebVectorTagger(pool, None, "q", RuntimeConfig()).config.thres == 0.55
    assert CelebVectorTagger(pool, None, "q", RuntimeConfig(ground_truth="iq__x")).config.thres == 0.4


def test_match_is_exact_argmax(pool_path):
    q = np.random.RandomState(0).randn(50, DIM).astype(np.float32)
    idx, scores = CelebPool(pool_path).match(q.copy())
    feats = np.stack([_unit(0), _unit(1)])
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    assert (idx == np.argmax(qn @ feats.T, axis=1)).all()
    assert np.allclose(scores, np.max(qn @ feats.T, axis=1), atol=1e-6)


def test_parse_binary_search():
    vecs = np.arange(6, dtype="<f4").reshape(2, 3)
    meta = {"count": 2, "dim": 3, "dtype": "float32", "byte_order": "little",
            "results": [{"vector": {"start_time": 0}}, {"vector": {"start_time": 1000}}]}
    body = (b"--B\r\nContent-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n"
            + json.dumps(meta).encode() + b"\n"
            + b"\r\n--B\r\nContent-Disposition: form-data; name=\"vectors\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            + vecs.tobytes() + b"\r\n--B--\r\n")
    got_meta, got = parse_binary_search("multipart/mixed; boundary=B", body)
    assert got_meta["results"][1]["vector"]["start_time"] == 1000
    assert np.array_equal(got, vecs)

    empty = dict(meta, count=0, results=[])
    body = (b"--B\r\nContent-Disposition: form-data; name=\"metadata\"\r\nContent-Type: application/json\r\n\r\n"
            + json.dumps(empty).encode() + b"\r\n--B\r\nContent-Disposition: form-data; name=\"vectors\"\r\n"
            + b"Content-Type: application/octet-stream\r\n\r\n\r\n--B--\r\n")
    assert parse_binary_search("multipart/mixed; boundary=B", body)[1].shape == (0, 3)
