from __future__ import annotations

import os
from dataclasses import asdict
from typing import List

import cv2
import numpy as np
import torch
from dacite import from_dict
from easydict import EasyDict as edict
from facenet_pytorch import MTCNN
from loguru import logger

from celeb import face_model

from common_ml.tagging.models.frame_based import BatchFrameModel
from common_ml.tagging.models.tag_types import FrameTag

from celeb_vector.config import RuntimeConfig

_MODEL_VERSION = "insightface-r100-ii" # additional info for every vector so the vector DB can validate / re-embed if the model changes
_IMAGE_SIZE = [112, 112]  # InsightFace r100 input from celeb/model.py


class CelebVectorizer(BatchFrameModel):
    """Detects faces and emits one embedding per face as a Tag with vector. 
    Handles the embedding (once) in the model-celeb pipeline so adding celebrities becomes fast matmul.
    (Pooling, thresholding, matching, clustering, etc. move to query time with the vector DB.)"""

    # frames per MTCNN forward pass (bounds GPU memory at high resolutions)
    DET_BATCH_SIZE = 32

    def __init__(self, model_input_path: str, cfg: RuntimeConfig) -> None:
        self.config = cfg
        self.model_input_path = model_input_path
        # torch 1.9 has no kernels for newer GPUs (e.g. L40S sm_89): cuda.is_available() is True 
        # but use the GPU only if its arch is in torch's build list, else fall back to CPU (matches mxnet MTCNN).
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm = f"sm_{major}{minor}"
            # CUDA binaries run on GPUs of the same major arch with an equal or newer minor
            # (e.g. torch's sm_70 build runs on sm_75 Turing), but not across major arches
            built = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
            if any(int(a[3:-1]) == major and int(a[-1]) <= minor for a in built):
                self.device = torch.device('cuda:0')
            else:
                logger.warning(f"GPU {sm} unsupported by this torch build {torch.cuda.get_arch_list()}; running on CPU")
        self.args = self._add_params()

        self.detector = MTCNN(image_size=self.args.image_size[0], keep_all=True, device=self.args.device)
        logger.info(f"MTCNN on GPU: {next(self.detector.parameters()).is_cuda}")
        self.model = face_model.FaceModel(self.args)

    def _add_params(self) -> edict:
        return edict({
            'image_size': _IMAGE_SIZE,
            'model': os.path.join(self.model_input_path, 'models/model-r100-ii/model,0'),
            'ga_model': '',
            # mxnet-cu101 supports the same (pre-Ampere) GPUs as torch 1.9, so follow the torch device check
            'gpu': 0 if self.device.type == 'cuda' else -1,
            'embed_batch_size': 32,
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

    def _detect(self, imgs: np.ndarray):
        """Batched MTCNN over (N, H, W, 3) frames; returns per-frame (boxes, probs)."""
        boxes, probs = [], []
        for s in range(0, len(imgs), self.DET_BATCH_SIZE):
            # landmarks (keypoints of eyes, nose, mouth) unused so not requested
            b, p = self.detector.detect(imgs[s:s + self.DET_BATCH_SIZE])
            boxes.extend(b)
            probs.extend(p)
        return boxes, probs

    def tag_frames(self, imgs: np.ndarray) -> List[List[FrameTag]]:
        """imgs: (N, H, W, 3) uint8 RGB. Per frame, one FrameTag per detected face:
        `vector`=the InsightFace embedding, `box`=the normalized box."""
        imgs = np.asarray(imgs)
        boxes_per_frame, probs_per_frame = self._detect(imgs)

        crops: List[np.ndarray] = []
        norm_boxes: List[List[float]] = []
        frame_idx: List[int] = []
        for i, (img, boxes, probs) in enumerate(zip(imgs, boxes_per_frame, probs_per_frame)):
            if boxes is None:
                continue
            h, w, _ = img.shape
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
                frame_idx.append(i)

        out: List[List[FrameTag]] = [[] for _ in range(len(imgs))]
        if not crops:
            return out

        # embed all faces of the batch at once; (N, len(vector)) matches celeb/model.py and InsightFace in face_model does L2-normalization
        feats = self.model.get_batch_features(aligned_batch=np.array(crops), batch_size=32)

        for vec, nb, i in zip(feats, norm_boxes, frame_idx):
            v = vec.astype(np.float32)
            # InsightFace already L2-normalizes; re-normalize after the float32 cast so the
            # emitted vector is exactly unit-length (cosine == dot for the vector DB).
            v = v / (np.linalg.norm(v) + 1e-12)
            out[i].append(FrameTag(
                tag="",
                vector=v.tolist(),
                box={"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                additional_info={"box": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]}},
            ))
        return out

    def tag_frame(self, img: np.ndarray) -> List[FrameTag]:
        return self.tag_frames(np.array([img]))[0]
