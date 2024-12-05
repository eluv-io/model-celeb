import yaml
import os
from typing import Any

def load_config() -> Any:   
    path = os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()