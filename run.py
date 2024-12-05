
import argparse
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from typing import List, Optional
from dataclasses import asdict, dataclass
from common_ml.utils import nested_update
from common_ml.types import Data

from celeb.model import CelebRecognition
from config import config

@dataclass
class RuntimeConfig(Data):
    fps: int
    thres: float
    ipt_rgb: bool
    allow_single_frame: bool
    content_id: Optional[str]=None

    class Schema(Schema):
        freq = fields.Int(required=True)
        thres = fields.Float(required=True)
        ipt_rgb = fields.Bool(required=True)
        allow_single_frame = fields.Bool(required=True)
        content_id = fields.Str(required=False)

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

def run(video_paths: List[str], runtime_config: str=None):
    files = video_paths
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        if runtime_config.endswith('.json'):
            with open(runtime_config, 'r') as fin:
                cfg = json.load(fin)
        else:
            cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    try:
        runtime_config = RuntimeConfig.from_dict(cfg)
    except ValidationError as e:
        logger.error("Received invalid runtime config.")
        raise e
    filedir = os.path.dirname(os.path.abspath(__file__))
    tags_out = os.getenv('TAGS_PATH', os.path.join(filedir, 'tags'))
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
    model = CelebRecognition(config["model_path"], thres=runtime_config.thres)
    if runtime_config.content_id is not None:
        model.set_content(runtime_config.content_id)
    for fname in files:
        logger.info(f"Tagging video: {fname}")
        ftags, vtags = model.tag_video(fname, runtime_config.allow_single_frame, runtime_config.fps)
        with open(os.path.join(tags_out, f"{os.path.basename(fname).split('.')[0]}_tags.json"), 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in vtags]))
        with open(os.path.join(tags_out, f"{os.path.basename(fname).split('.')[0]}_frametags.json"), 'w') as fout:
            ftags = {k: [asdict(tag) for tag in v] for k, v in ftags.items()}
            fout.write(json.dumps(ftags))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.video_paths, args.config)