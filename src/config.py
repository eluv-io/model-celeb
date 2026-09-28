import os
from typing import Any

import yaml


def load_config(path: str = None) -> Any:
    """Loads a container's config.yml (CONFIG_PATH, else ./config.yml). Relative `storage`
    paths are made absolute against the file's directory."""
    path = path or os.getenv('CONFIG_PATH', 'config.yml')
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    for key in config.get('storage', {}):
        if not config['storage'][key].startswith('/'):
            config['storage'][key] = os.path.join(os.path.dirname(path), config['storage'][key])
    return config
