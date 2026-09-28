from __future__ import annotations

import json
from dataclasses import dataclass, field
from email import message_from_bytes
from email.policy import HTTP
from typing import Dict, List, Optional

import numpy as np
import requests
from loguru import logger


@dataclass
class StoredVector:
    """One vector hit from the vectorstore, times already aligned to the full content."""
    start_time: int
    end_time: int
    frame_idx: Optional[int]
    additional_info: Dict = field(default_factory=dict)
    source: str = ""


class VectorstoreClient:
    """Reads the face vectors that model-celeb-vector wrote to an index."""

    def __init__(self, base_url: str, index_qid: str, token: str, timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.index_qid = index_qid
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json', 'Authorization': f"Bearer {token}"})

    def get_vectors(
        self,
        content_qid: str,
        track: str,
        start_time_gte: int,
        start_time_lte: int,
    ) -> tuple[List[StoredVector], np.ndarray]:
        """Returns every vector of `content_qid` on `track` starting in [start_time_gte, start_time_lte] (ms),
        as (metadata, embeddings of shape (N, dim)), via /search/binary to avoid shipping floats as JSON."""
        body = {
            'qids': [content_qid],
            'track': track,
            'start_time_gte': start_time_gte,
            'start_time_lte': start_time_lte,
            'include_vector': True,
            'limit': -1,  # no limit
        }
        url = f"{self.base_url}/indexes/{self.index_qid}/search/binary"
        response = self.session.post(url, data=json.dumps(body), timeout=self.timeout)
        if not response.ok:
            raise RuntimeError(f"vectorstore search failed ({response.status_code}) for index {self.index_qid}: {response.text[:1000]}")

        meta, vectors = parse_binary_search(response.headers.get('Content-Type', ''), response.content)
        hits = [
            StoredVector(
                start_time=r['vector'].get('start_time', 0),
                end_time=r['vector'].get('end_time', 0),
                frame_idx=r['vector'].get('frame_idx'),
                additional_info=r['vector'].get('additional_info') or {},
                source=r['vector'].get('source', ''),
            )
            for r in meta.get('results') or []
        ]
        if len(hits) != len(vectors):
            raise RuntimeError(f"vectorstore returned {len(hits)} results but {len(vectors)} vectors")
        logger.info(f"fetched {len(hits)} vectors from index {self.index_qid} for {content_qid} [{start_time_gte}, {start_time_lte}]")
        return hits, vectors


def parse_binary_search(content_type: str, body: bytes) -> tuple[dict, np.ndarray]:
    """Splits a /search/binary multipart/mixed response into (metadata json, (count, dim) float32 array)."""
    if not content_type.startswith('multipart/'):
        raise RuntimeError(f"expected a multipart response from /search/binary, got {content_type!r}")
    msg = message_from_bytes(b'Content-Type: ' + content_type.encode() + b'\r\n\r\n' + body, policy=HTTP)

    meta, raw = None, None
    for part in msg.iter_parts():
        name = part.get_param('name', header='content-disposition')
        ctype = part.get_content_type()
        if name == 'metadata' or ctype == 'application/json':
            meta = json.loads(part.get_payload(decode=True))
        elif name == 'vectors' or ctype == 'application/octet-stream':
            raw = part.get_payload(decode=True) or b''
    if meta is None:
        raise RuntimeError("/search/binary response has no metadata part")

    count = int(meta.get('count', 0))
    if count == 0:
        return meta, np.zeros((0, int(meta.get('dim') or 0)), dtype=np.float32)
    if raw is None:
        raise RuntimeError("/search/binary response has no vectors part")

    dtype = np.dtype(meta.get('dtype') or 'float32')
    dtype = dtype.newbyteorder('>' if meta.get('byte_order') == 'big' else '<')
    vectors = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(count, int(meta['dim']))
    return meta, vectors
