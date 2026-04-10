
import os

from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from celeb.ground_truth import GroundTruthFetcher
from celeb.model import CelebRecognition
from celeb.config import RuntimeConfig
from config import config

if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb')

    catch_errors()

    params = get_params()

    params = from_dict(RuntimeConfig, data=params)

    token = os.getenv("ELV_TOKEN")
    if token is None:
        raise ValueError("ELV_TOKEN environment variable not set")

    fetcher = GroundTruthFetcher(url=config["ground_truth"]["url"], token=token, base_dir=config["container"]["gt_path"])

    # resolve gt pool
    pool = fetcher.fetch(params.ground_truth)

    model = CelebRecognition(model_input_path=config["container"]["model_path"], pool_path=pool, cfg=params)

    run_default(model)