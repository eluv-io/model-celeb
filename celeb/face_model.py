from __future__ import absolute_import
from __future__ import division

from loguru import logger as logging
from typing import List

import numpy as np
import mxnet as mx
import cv2
import sklearn.preprocessing
import torch
from facenet_pytorch import InceptionResnetV1


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    logging.info(f'loading {prefix} {epoch}')
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, args):
        self.args = args
        if args.gpu >= 0:
            ctx = mx.gpu(args.gpu)
        else:
            ctx = mx.cpu()

        self.image_size = args.image_size
        self.model = None
        self.ga_model = None
        self.model_vgg = None
        self.content_type = args.content_type

        if self.content_type in ['image']:
            self.model_vgg = InceptionResnetV1(
                pretrained='vggface2').to(args.device).eval()
            logging.info(f'loading vggface2')
        else:
            self.model = get_model(ctx, self.image_size, args.model, 'fc1')
            logging.info(f'loading insightface')
        if len(args.ga_model) > 0:
            self.ga_model = get_model(
                ctx, self.image_size, args.ga_model, 'fc1')

        self.threshold = args.threshold
        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        # self.det_factor = 0.9

    def get_feature(self, aligned):
        """
        Args:
            aligned: (1, 3, image_size[0], image_size[1]))
        """
        input_blob = aligned
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        logging.debug(f'embedding shape {embedding.shape}')
        embedding = sklearn.preprocessing.normalize(
            embedding, axis=1)  # .flatten()
        return embedding

    def get_batch_features(self, aligned_batch, batch_size=32):
        """
        Args:
            aligned_batch: Batch of aligned images with shape (n_images, 3, image_size[0], image_size[1])
            batch_size: Number of images to process in a single pass
        """
        if self.model_vgg is not None:
            logging.info(
                f'inference with vggface2 for content type {self.content_type}')
            return self.get_batch_feature_vgg(aligned_batch, batch_size)
        if self.model is not None:
            logging.info(f'inference with insight face')
            return self.get_batch_feature_insight(aligned_batch, batch_size)
        else:
            raise ValueError('No model loaded for feature extraction')

    def get_batch_feature_insight(self, aligned_batch, batch_size):
        n_images = len(aligned_batch)
        embeddings = []

        for start_idx in range(0, n_images, batch_size):
            end_idx = min(start_idx + batch_size, n_images)
            input_blob = aligned_batch[start_idx:end_idx]
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.model.forward(db, is_train=False)
            embedding = self.model.get_outputs()[0].asnumpy()
            logging.info(f'embedding shape {embedding.shape}')
            embedding = sklearn.preprocessing.normalize(embedding, axis=1)
            embeddings.append(embedding)

        embeddings = np.vstack(embeddings)

        return embeddings

    def get_batch_feature_vgg(self, aligned_batch: List[np.ndarray], batch_size: int) -> np.ndarray:
        embeddings = []

        imgs_tensor = torch.tensor(aligned_batch, dtype=torch.float32) / 255.0
        device = next(self.model_vgg.parameters()).device
        imgs_tensor = imgs_tensor.to(device)

        self.model_vgg.eval()
        with torch.no_grad():
            for i in range(0, len(imgs_tensor), batch_size):
                batch = imgs_tensor[i:i + batch_size]
                emb = self.model_vgg(batch)
                embeddings.append(emb)

        return torch.cat(embeddings, dim=0).detach().cpu().numpy()

    def get_ga(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age
