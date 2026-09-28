from __future__ import annotations

import mxnet as mx
import numpy as np
from loguru import logger


class InsightFaceEmbedder:
    """InsightFace r100 (mxnet) embeddings for aligned (N, 3, 112, 112) uint8 face crops."""

    def __init__(self, model_prefix: str, epoch: int = 0, gpu: int = -1, image_size=(112, 112), batch_size: int = 32):
        self.batch_size = batch_size
        ctx = mx.gpu(gpu) if gpu >= 0 else mx.cpu()
        # on GPU, bind at a fixed batch size and pad partial batches so the executor
        # (and cudnn autotuning) is not rebuilt for every new batch shape
        self.fixed_batch_size = batch_size if gpu >= 0 else None

        logger.info(f'loading insightface {model_prefix} {epoch} on {ctx}')
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
        sym = sym.get_internals()['fc1_output']
        self.model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.model.bind(data_shapes=[('data', (self.fixed_batch_size or 1, 3, image_size[0], image_size[1]))])
        self.model.set_params(arg_params, aux_params)

    def embed(self, faces: np.ndarray) -> np.ndarray:
        """faces: (N, 3, H, W) uint8. Returns (N, 512) float32, L2-normalized (cosine == dot)."""
        embeddings = []
        for s in range(0, len(faces), self.batch_size):
            batch = faces[s:s + self.batch_size]
            n = len(batch)
            if self.fixed_batch_size and n < self.fixed_batch_size:
                pad = np.zeros((self.fixed_batch_size - n,) + batch.shape[1:], dtype=batch.dtype)
                batch = np.concatenate([batch, pad])
            self.model.forward(mx.io.DataBatch(data=(mx.nd.array(batch),)), is_train=False)
            embeddings.append(self.model.get_outputs()[0].asnumpy()[:n])
        emb = np.vstack(embeddings).astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
