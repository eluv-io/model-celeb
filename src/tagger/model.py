from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
from dacite import from_dict
from loguru import logger

from common_ml.tagging.messages import FrameInfo, Tag
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
        frame_tags = [self._frame_tag(hit, name, score, fpath) for hit, name, score in faces]
        return frame_tags + self._combine_adjacent(frame_tags, interval_ms, _infer_frame_ms(hits))

    def _frame_tag(self, hit: StoredVector, name: str, score: float, fpath: str) -> Tag:
        frame_info = None
        if hit.frame_idx is not None:
            frame_info = FrameInfo(frame_idx=hit.frame_idx, box=hit.additional_info.get("box", {}))
        return Tag(
            start_time=hit.start_time,
            end_time=hit.end_time,
            tag=name,
            source_media=fpath,
            additional_info={"confidence": round(score, 4)},
            frame_info=frame_info,
        )

    def _combine_adjacent(self, frame_tags: List[Tag], interval_ms: Optional[int], frame_ms: int) -> List[Tag]:
        """Segment tags from frame tags, as common-ml's AVModel.from_frame_model does for frame models:
        one tag per run of consecutive sampled frames showing the same name, ending one video frame
        after the run's last frame. Runs of a single frame are dropped unless allow_single_frame."""
        def next_sample(prev: Tag, t: Tag) -> bool:
            # the sampled frame right after prev's: a gap of about one sample interval
            return bool(interval_ms) and round((t.start_time - prev.start_time) / interval_ms) == 1

        by_name: Dict[str, Dict[int, Tag]] = {}
        for t in frame_tags:
            # several faces with the same name on one frame count once
            by_name.setdefault(t.tag, {}).setdefault(t.start_time, t)

        def combined(left: Tag, right: Tag) -> Tag:
            return Tag(
                start_time=left.start_time,
                end_time=right.end_time + frame_ms,
                tag=left.tag,
                source_media=left.source_media,
                track=left.track,
            )

        out = []
        for items in by_name.values():
            run = [items[p] for p in sorted(items)]
            left = right = run[0]
            for item in run[1:] + [None]:
                if item is not None and next_sample(right, item):
                    right = item
                    continue
                if self.config.allow_single_frame or right is not left:
                    out.append(combined(left, right))
                left = right = item
        return sorted(out, key=lambda t: (t.start_time, t.tag))


def _infer_sample_interval(hits: List[StoredVector]) -> Optional[int]:
    """The smallest gap between distinct face timestamps, i.e. the embedding model's frame spacing."""
    times = np.unique([h.start_time for h in hits])
    gaps = np.diff(times)
    gaps = gaps[gaps > 0]
    return int(gaps.min()) if len(gaps) else None


def _infer_frame_ms(hits: List[StoredVector]) -> int:
    """The duration (ms) of one video frame, from the content aligned frame index and start time
    of the latest frame; 0 if no hit has a frame index."""
    known = [h for h in hits if h.frame_idx]
    if not known:
        return 0
    last = max(known, key=lambda h: h.frame_idx)
    return round(last.start_time / last.frame_idx)
