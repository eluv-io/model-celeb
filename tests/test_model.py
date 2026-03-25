import os

from celeb.model import CelebRecognition, RuntimeConfig

from common_ml.tagging.file_tagger import FileTagger

from config import config

TEST_FILE = os.path.join(os.path.dirname(__file__), "../test-files/1.mp4")

def test_model():
    model = CelebRecognition(model_input_path=config["container"]["model_path"], cfg=RuntimeConfig())
    tagger = FileTagger.from_frame_model(model)
    tags = tagger.tag(TEST_FILE)
    assert len(tags) > 0
    for tag in tags:
        assert tag.source_media == TEST_FILE
        assert tag.end_time > tag.start_time or tag.frame_info
        if tag.frame_info:
            assert tag.frame_info.box