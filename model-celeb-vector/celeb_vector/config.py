from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """Runtime tunables for the celeb face-embedding tagger, injected per-request via
    `--params` in run.py.
    This model only detects faces and emits their embeddings."""

    # Compared to model-celeb: no similarity threshold, ground_truth pool, content_id, or cast list.
    # Standardize image and video embeddings using just the video path model InsightFace.

    # Drop detections whose normalized box area is below this (same knob as model-celeb).
    min_box_size: float = 0.0

    # MTCNN confidence gate. Hardcoded to 0.96 in model-celeb's detect_batch.
    det_confidence: float = 0.96

    # InsightFace already L2-normalizes each emitted face vector (expected by vector DB)
    # so cosine similarity reduces to a dot product.

    # identity matching (dot-product against celebrity pool) and clustering happen in vector search
