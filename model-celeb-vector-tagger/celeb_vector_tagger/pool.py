from __future__ import annotations

import json
import os
from typing import List, Optional, Set, Tuple

import numpy as np
from loguru import logger


class CelebPool:
    """A ground truth pool (feats.npy / gt.npy / id2name.json, same layout model-celeb uses)
    that assigns each face embedding its most similar pool face."""

    # pool rows scored at a time, bounds the (rows, queries) similarity matrix
    CHUNK = 131072

    def __init__(self, pool_path: str):
        self.pool_path = pool_path
        # upcast once (as model-celeb does) so every search is a plain fp32 matmul
        self.feats = np.load(os.path.join(pool_path, 'feats.npy')).astype(np.float32)
        self.gt = np.load(os.path.join(pool_path, 'gt.npy'))
        with open(os.path.join(pool_path, 'id2name.json'), 'r') as f:
            self.id2name = json.load(f)
        logger.info(f"loaded pool {pool_path}: {len(self.feats)} faces, {len(self.id2name)} identities")

        self.cast_check = {}
        ca_path = os.path.join(pool_path, 'ca_lookup.json')
        if os.path.exists(ca_path):
            with open(ca_path, 'r') as f:
                self.cast_check = {k: set(v) if v else None for k, v in json.load(f).items()}

    def cast_pool(self, content_id: Optional[str], restrict_list: Optional[List[str]]) -> Optional[Set[str]]:
        """Names allowed for this content, None if unrestricted. Same precedence as model-celeb."""
        if content_id:
            return self.cast_check.get(content_id, None)
        if restrict_list:
            return set(restrict_list)
        restrict_path = os.path.join(self.pool_path, 'restrict.txt')
        if os.path.exists(restrict_path):
            with open(restrict_path, 'r') as f:
                return {line.strip() for line in f if line.strip()}
        return None

    def name(self, idx: int) -> Optional[str]:
        return self.id2name.get(str(self.gt[idx]))

    def match(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """queries: (N, dim) embeddings. Returns (pool index, cosine similarity) of the best pool face per query."""
        queries = queries.astype(np.float32)
        # stored vectors are halfvec, so re-normalize before comparing against the (unit) pool
        queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
        best_idx = np.zeros(len(queries), dtype=np.int64)
        best = np.full(len(queries), -np.inf, dtype=np.float32)
        if len(queries) == 0:
            return best_idx, best
        rows = np.arange(len(queries))
        for s in range(0, len(self.feats), self.CHUNK):
            # (queries, pool rows) so the argmax runs along contiguous memory
            sims = queries @ self.feats[s:s + self.CHUNK].T
            idx = np.argmax(sims, axis=1)
            val = sims[rows, idx]
            better = val > best
            best[better] = val[better]
            best_idx[better] = idx[better] + s
        return best_idx, best
