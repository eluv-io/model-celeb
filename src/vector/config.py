from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Runtime tunables for the celeb face-embedding tagger, injected per-request via
    `--params` in run.py.
    This model only detects faces and emits their embeddings."""

    # Naming faces (threshold, ground truth pool, cast list) happens later in src.tagger.

    # Drop detections whose normalized box area is below this.
    min_box_size: float = 0.1

    # MTCNN confidence gate.
    det_confidence: float = 0.96
