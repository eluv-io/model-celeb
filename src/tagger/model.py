from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
from dacite import from_dict
from loguru import logger

from common_ml.tagging.messages import Tag
from common_ml.tagging.models.av import AVModel

from src.tagger.config import RuntimeConfig
from src.tagger.pool import CelebPool
from src.tagger.vectorstore import StoredVector, VectorstoreClient


class CelebVectorTagger(AVModel):
    """Names the faces model-celeb-vector detected and embedded into the vectorstore, by reading
    those embeddings back and matching them against a ground truth pool.

    Runs as a content aligned processor: each input file is a JSON time range
    ({"start_time": ms, "end_time": ms}) of the content, and every face vector starting in
    that range is matched and emitted as tags on the same content aligned timeline."""

    def __init__(self, pool: CelebPool, vectorstore: VectorstoreClient, content_qid: str, cfg: RuntimeConfig) -> None:
        self.pool = pool
        self.vectorstore = vectorstore
        self.content_qid = content_qid
        self.config = cfg
        if self.config.thres == -1:
            self.config.thres = 0.55 if self.config.ground_truth == "IBC" else 0.4

    def set_config(self, config: dict) -> None:
        self.config = from_dict(RuntimeConfig, config)

    def get_config(self) -> dict:
        return asdict(self.config)

    def tag(self, fpath: str) -> List[Tag]:
        with open(fpath, 'r') as f:
            interval = json.load(f)
        start, end = int(interval['start_time']), int(interval['end_time'])

        # intervals are [start, end): a face at exactly `end` belongs to the next one
        hits, vectors = self.vectorstore.get_vectors(
            content_qid=self.content_qid, track=self.config.vector_track,
            start_time_gte=start, start_time_lte=end - 1)
        if not hits:
            return []

        pool_idx, scores = self.pool.match(vectors)
        cast_pool = self.pool.cast_pool(self.config.content_id, self.config.restrict_list)

        faces = []  # (hit, name, score) for faces that pass the threshold and cast filter
        for hit, idx, score in zip(hits, pool_idx, scores):
            if score < self.config.thres:
                continue
            name = self.pool.name(int(idx))
            if name is None or (cast_pool is not None and name not in cast_pool):
                continue
            faces.append((hit, name, float(score)))
        logger.info(f"[{start}, {end}): {len(faces)}/{len(hits)} faces above {self.config.thres}")

        interval_ms = self.config.sample_interval_ms or _infer_sample_interval(hits)
        frame_tags = [self._face_tag(hit, name, score, fpath) for hit, name, score in faces]
        return frame_tags + self._merge_adjacent(faces, interval_ms, fpath)

    def _face_tag(self, hit: StoredVector, name: str, score: float, fpath: str) -> Tag:
        info: Dict = {"confidence": round(score, 4)}
        if "box" in hit.additional_info:
            info["box"] = hit.additional_info["box"]
        if hit.frame_idx is not None:
            info["frame_idx"] = hit.frame_idx
        return Tag(
            start_time=hit.start_time,
            end_time=hit.end_time,
            tag=name,
            source_media=fpath,
            additional_info=info,
        )

    def _merge_adjacent(self, faces, interval_ms: Optional[int], fpath: str) -> List[Tag]:
        """One tag per run of sampled frames that show the same name, where consecutive frames of a
        run are at most ~one sample interval apart. Each run extends one interval past its last frame."""
        times_by_name: Dict[str, List[int]] = {}
        for hit, name, _ in faces:
            times_by_name.setdefault(name, []).append(hit.start_time)

        out = []
        for name, times in times_by_name.items():
            times = sorted(set(times))
            run_start = prev = times[0]
            for t in times[1:] + [None]:
                if t is not None and interval_ms is not None and t - prev <= 1.5 * interval_ms:
                    prev = t
                    continue
                out.append(Tag(
                    start_time=run_start,
                    end_time=prev + (interval_ms or 0),
                    tag=name,
                    source_media=fpath,
                ))
                if t is not None:
                    run_start = prev = t
        return sorted(out, key=lambda t: (t.start_time, t.tag))


def _infer_sample_interval(hits: List[StoredVector]) -> Optional[int]:
    """The smallest gap between distinct face timestamps, i.e. the embedding model's frame spacing."""
    times = np.unique([h.start_time for h in hits])
    gaps = np.diff(times)
    gaps = gaps[gaps > 0]
    return int(gaps.min()) if len(gaps) else None
