
import argparse
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from typing import List, Optional
from common_ml.utils import nested_update
from common_ml.model import default_tag
import setproctitle

from celeb.model import CelebRecognition
from config import config

# Generate tag files from a list of video/image files and a runtime config
# Runtime config follows the schema found in celeb.model.RuntimeConfig
def run(file_paths: List[str], runtime_config: str=None):
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    logger.debug("config:\n" + json.dumps(cfg, indent = 2))
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    model = CelebRecognition(config["container"]["model_path"], runtime_config=cfg)
    default_tag(model, file_paths, out_path)
        
if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb')
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.file_paths, args.config)
