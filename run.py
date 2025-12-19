import argparse
import os
import json
import sys
from typing import List, Optional
from dacite import from_dict
import setproctitle

from common_ml.utils import nested_update
from common_ml.model import default_tag, run_live_mode

from celeb.model import CelebRecognition
from celeb.config import RuntimeConfig
from config import config

def get_runtime_config(runtime_config: Optional[str]) -> RuntimeConfig:
    """Get the runtime configuration, merging with defaults if provided"""
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        cfg = nested_update(config["runtime"]["default"], cfg)
    
    return from_dict(RuntimeConfig, cfg)

# Generate tag files from a list of video/image files and a runtime config
# Runtime config follows the schema found in celeb.config.RuntimeConfig
def run(file_paths: List[str], cfg: RuntimeConfig):
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    model = CelebRecognition(config["container"]["model_path"], cfg=cfg)
    default_tag(model, file_paths, out_path)

def get_tag_fn(runtime_config: Optional[str]):
    """Create a tag function with the specified configuration"""
    cfg = get_runtime_config(runtime_config)
    
    def tag_fn(file_paths: List[str]):
        run(file_paths, cfg)
    
    return tag_fn

if __name__ == '__main__':
    setproctitle.setproctitle('model-celeb')
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths', default=[])
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    args = parser.parse_args()

    tag_fn = get_tag_fn(args.config)
    
    if args.live:
        print('Running in live mode...')
        run_live_mode(tag_fn)
    else:
        if not args.file_paths:
            print("Error: No file paths provided")
            sys.exit(1)
        print('Running in batch mode')
        tag_fn(args.file_paths)