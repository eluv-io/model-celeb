import io
import os
import tempfile
import zipfile
from loguru import logger

import requests

class GroundTruthFetcher:
    def __init__(
        self, 
        url: str, 
        token: str,
        base_dir: str
    ):
        self.url = url
        self.token = token
        self.base_dir = base_dir

    def need_fetch(self, gt: str) -> str:
        """Returns if ground truth is needed to be fetched"""
        local_path = os.path.join(self.base_dir, gt)
        if os.path.exists(local_path):
            logger.info(f"Ground truth {gt} already exists at {local_path}, skipping download")
            return False
        elif not gt.startswith("iq__"):
            raise ValueError(f"Ground truth identifier {gt} is not valid - Should be a content id")
        else:
            return True

    def fetch(self, gt: str) -> str:
        """Returns local path for gt, downloading and unpacking zip if it doesn't exist."""
        if not self.need_fetch(gt):
            return os.path.join(self.base_dir, gt)
        elif not gt.startswith("iq__"):
            raise ValueError(f"Ground truth identifier {gt} is not valid - Should be a content id")

        logger.info(f"Fetching ground truth {gt} from {self.url}")
        response = requests.get("/".join([self.url, gt, "pool"]), headers={"Authorization":self.token})
        response.raise_for_status()

        out_path = tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(out_path)
        logger.info(f"Extracted celeb pool to {out_path}")
        
        return out_path
