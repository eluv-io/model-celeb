import os

import numpy as np

from src.vector import CelebVectorizer, RuntimeConfig

from common_ml.tagging.file_tagger import FileTagger

REPO = os.path.join(os.path.dirname(__file__), "..")
TEST_FILE = os.path.join(REPO, "test-files/1.mp4")


def test_model():
    # needs the InsightFace weights under models/models (./pull-models or build.sh)
    model = CelebVectorizer(model_input_path=os.path.join(REPO, "models"), cfg=RuntimeConfig(min_box_size=0))
    tagger = FileTagger.from_frame_model(model)
    tags = tagger.tag(TEST_FILE)
    assert len(tags) > 0
    for tag in tags:
        assert tag.source_media == TEST_FILE
        # every tag carries a face embedding, not a name
        assert tag.tag == ""
        assert tag.vector is not None
        assert len(tag.vector) == 512
        # unit-normalized so cosine similarity reduces to a dot product downstream
        assert abs(np.linalg.norm(tag.vector) - 1.0) < 1e-3
        # vector tags pass through per-frame (AVModel.from_frame_model does not run-length merge them)
        if tag.frame_info:
            assert tag.frame_info.box

class _FakeDetector:
    def __init__(self, boxes, probs):
        self.boxes, self.probs = boxes, probs

    def detect(self, imgs):
        return [np.array(self.boxes, dtype=np.float32)] * len(imgs), [np.array(self.probs)] * len(imgs)


class _FakeEmbedder:
    def embed(self, faces):
        return np.ones((len(faces), 512), dtype=np.float32) / np.sqrt(512)


def _vectorizer(boxes, probs, **cfg):
    # skip __init__ (weights, GPU): only the detection filtering is under test
    model = CelebVectorizer.__new__(CelebVectorizer)
    model.config = RuntimeConfig(min_box_size=0, **cfg)
    model.detector = _FakeDetector(boxes, probs)
    model.embedder = _FakeEmbedder()
    return model


def test_max_faces_keeps_largest_in_detection_order():
    # 100x100 frame; face i is a square of side sides[i]
    sides = [10, 50, 20, 40, 30, 5]
    boxes = [[0, 0, s, s] for s in sides]
    img = np.zeros((1, 100, 100, 3), dtype=np.uint8)

    tags = _vectorizer(boxes, [0.99] * 6).tag_frames(img)[0]
    assert [t.box["x2"] for t in tags] == [0.5, 0.2, 0.4, 0.3]  # 4 largest, detection order kept

    tags = _vectorizer(boxes, [0.99] * 6, max_faces=2).tag_frames(img)[0]
    assert [t.box["x2"] for t in tags] == [0.5, 0.4]

    # 0 disables the limit; low-confidence faces are dropped before choosing the largest
    probs = [0.99, 0.5, 0.99, 0.99, 0.99, 0.99]
    assert len(_vectorizer(boxes, probs, max_faces=0).tag_frames(img)[0]) == 5
    assert [t.box["x2"] for t in _vectorizer(boxes, probs).tag_frames(img)[0]] == [0.1, 0.2, 0.4, 0.3]
