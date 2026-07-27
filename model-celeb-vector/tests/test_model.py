import os

import numpy as np

from celeb_vector.model import CelebVectorizer
from celeb_vector.config import RuntimeConfig

from common_ml.tagging.file_tagger import FileTagger

from config import config

# test-files live in the parent model-celeb/ repo, two levels up from this test.
TEST_FILE = os.path.join(os.path.dirname(__file__), "../../test-files/1.mp4")


def test_model():
    model = CelebVectorizer(model_input_path=config["container"]["model_path"], cfg=RuntimeConfig())
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