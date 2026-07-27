from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import List

import cv2
import numpy as np
import torch
from dacite import from_dict
from easydict import EasyDict as edict
from facenet_pytorch import MTCNN
from loguru import logger

# model-celeb-vector is nested under model-celeb, whose `celeb` package sits three levels
# up from this file (locally). (In the container both packages sit side by side under the workdir.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from celeb import face_model

from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.models.tag_types import FrameTag

from celeb_vector.config import RuntimeConfig

_MODEL_VERSION = "insightface-r100-ii" # additional info for every vector so the vector DB can validate / re-embed if the model changes
_IMAGE_SIZE = [112, 112]  # InsightFace r100 input from celeb/model.py


class CelebVectorizer(FrameModel):
    """Detects faces and emits one embedding per face as a Tag with vector. 
    Handles the embedding (once) in the model-celeb pipeline so adding celebrities becomes fast matmul.
    (Pooling, thresholding, matching, clustering, etc. move to query time with the vector DB.)"""

    def __init__(self, model_input_path: str, cfg: RuntimeConfig) -> None:
        self.config = cfg
        self.model_input_path = model_input_path
        # torch 1.9 has no kernels for newer GPUs (e.g. L40S sm_89): cuda.is_available() is True 
        # but use the GPU only if its arch is in torch's build list, else fall back to CPU (matches mxnet MTCNN).
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm = f"sm_{major}{minor}"
            if sm in torch.cuda.get_arch_list():
                self.device = torch.device('cuda:0')
            else:
                logger.warning(f"GPU {sm} unsupported by this torch build {torch.cuda.get_arch_list()}; detecting on CPU")
        self.args = self._add_params()

        self.detector = MTCNN(image_size=self.args.image_size[0], keep_all=True, device=self.args.device)
        logger.info(f"MTCNN on GPU: {next(self.detector.parameters()).is_cuda}")
        self.model = face_model.FaceModel(self.args)

    def _add_params(self) -> edict:
        return edict({
            'image_size': _IMAGE_SIZE,
            'model': os.path.join(self.model_input_path, 'models/model-r100-ii/model,0'),
            'ga_model': '',
            'gpu': -1, # keep mxnet on CPU to match celeb/model.py
            'threshold': 1.24,
            'content_type': 'video', # only use the InsightFace backend (video input embedding path in celeb/model.py)
            'device': self.device,
        })

    def set_config(self, config: dict) -> None:
        self.config = from_dict(RuntimeConfig, config)

    def get_config(self) -> dict:
        return asdict(self.config)

    @staticmethod
    def _box_area(box: List[float]) -> float: # matches celeb/model.py's _box_size
        return abs(box[2] - box[0]) * abs(box[3] - box[1])

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        """img: (H, W, 3) uint8 RGB. One FrameTag per detected face: 
        `vector`=the InsightFace embedding, `box`=the normalized box."""
        h, w, _ = img.shape
        boxes, probs = self.detector.detect(img)  # landmarks (keypoints of eyes, nose, mouth) unused so not requested
        if boxes is None:
            return []

        crops: List[np.ndarray] = []
        norm_boxes: List[List[float]] = []
        for box, prob in zip(boxes, probs):
            if prob is None or prob < self.config.det_confidence:
                continue
            nb = [round(float(box[0] / w), 4), round(float(box[1] / h), 4),
                  round(float(box[2] / w), 4), round(float(box[3] / h), 4)]
            if self._box_area(nb) < self.config.min_box_size:
                logger.debug("Face too small, skipping")
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            face = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if face.size == 0:
                continue
            c = cv2.resize(face, tuple(self.args.image_size))
            c = np.transpose(c, (2, 0, 1))  # H, W, C -> C, H, W, matches model-celeb preprocessing
            crops.append(c)
            norm_boxes.append(nb)

        if not crops:
            return []

        feats = self.model.get_batch_features(aligned_batch=np.array(crops), batch_size=32)  # (N, len(vector)) matches celeb/model.py and InsightFace in face_model does L2-normalization

        out: List[FrameTag] = []
        for vec, nb in zip(feats, norm_boxes):
            v = vec.astype(np.float32)
            # InsightFace already L2-normalizes; re-normalize after the float32 cast so the
            # emitted vector is exactly unit-length (cosine == dot for the vector DB).
            v = v / (np.linalg.norm(v) + 1e-12)
            out.append(FrameTag(
                tag="",  # resolve identity downstream against the pool (?)
                vector=v.tolist(),
                box={"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                additional_info={"model": _MODEL_VERSION, "dim": int(v.shape[0])}, # vector metadata for validation
            ))
        return out
