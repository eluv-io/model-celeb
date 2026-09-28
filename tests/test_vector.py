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