from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RuntimeConfig:
    """Runtime tunables, injected per-request via `--params` in run.py."""

    # Minimum similarity to the best matching pool face for a face to be tagged.
    # -1 picks the default for the pool: 0.55 for IBC, 0.4 otherwise.
    thres: float = -1

    # Ground truth pool to match against: a pool bundled under models/image_features
    # (e.g. IBC) or a content id to download with ELV_TOKEN.
    ground_truth: str = "IBC"

    # Restrict tags to a cast: the pool's ca_lookup entry
    # for content_id, else restrict_list, else the pool's restrict.txt.
    content_id: Optional[str] = None
    restrict_list: Optional[List[str]] = None

    # Vectorstore track model-celeb-vector's embeddings were written to.
    vector_track: str = "face_vectors"

    # Spacing (ms) between the frames model-celeb-vector sampled; same-name faces on
    # neighbouring sampled frames merge into one tag. None infers it from the vectors.
    sample_interval_ms: Optional[int] = None
