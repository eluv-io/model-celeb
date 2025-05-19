import os
import time
import json
from collections import defaultdict
from typing import List, Optional, Union, Tuple

import networkx as nx
from easydict import EasyDict as edict
from loguru import logger
import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from facenet_pytorch import MTCNN
from dataclasses import dataclass, asdict
from marshmallow import Schema, fields

from . import face_model
from common_ml.tags import FrameTag
from common_ml.model import FrameModel
from common_ml.types import Data
from config import config

@dataclass
class RuntimeConfig(Data):
    fps: int
    thres: float
    thres_cast: float
    min_box_size: float
    ipt_rgb: bool
    allow_single_frame: bool
    ground_truth: str
    content_id: Optional[str]=None
    restrict_list: Optional[List[str]]=None

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

class CelebRecognition(FrameModel):
    def __init__(self, model_input_path: str, runtime_config: Union[dict, RuntimeConfig]) -> None:
        if isinstance(config, dict):
            self.config = RuntimeConfig.from_dict(runtime_config)
        else:
            self.config = runtime_config
        self.model_input_path = model_input_path
        self.pool_path = os.path.join(config["container"]["gt_path"], self.config.ground_truth)
        self.args = self._add_params()
        # self.detector = cv2.dnn.readNetFromCaffe(
        #    self.args.res10ssd_prototxt_path, self.args.res10ssd_model_path)
        self.detector = MTCNN(keep_all=True, device=torch.device("cuda:0"))
        logger.info(
            f"MTCNN parameters stored on GPU: {next(self.detector.parameters()).is_cuda}")
        self.model = face_model.FaceModel(self.args)
        
        im_pool_feats = np.load(self.args.im_pool_feats)
        self.im_pool_feats = im_pool_feats.astype(np.float32)
        self.gt = np.load(self.args.gt)
        self.logged = False
        with open(self.args.id2name, 'r') as f:
            self.id2name = json.load(f)         
            self._create_name2id()
        with open(self.args.cast_check, 'r') as f:
            self.cast_check = json.load(f)
            self.cast_check = {
                k: set(v) if v else None for k, v in self.cast_check.items()}
            
    def _create_name2id(self):
        rev = defaultdict(list)
        for id, name in self.id2name.items():
            rev[name].append(id)
        self.name2id = rev
        
    def _add_params(self):
        io_path = self.model_input_path
        gt_path = self.pool_path
        params = edict({

            'image_size': '112,112',
            # 'path to load model'
            'model': os.path.join(io_path, 'models/model-r100-ii/model,0'),
            'ga_model': '',  # 'path to load model'
            'gpu': -1,  # 'gpu id'
            'det': 0,  # 'mtcnn option, 1 means using R+O, 0 means detect from begining'
            'flip_celeb': 0,  # 'whether do lr flip aug'
            'threshold': 1.24,  # 'ver dist threshold'
            # 'image_features/feats_with_ibc.npy'
            'im_pool_feats': os.path.join(gt_path, 'feats.npy'),
            # 'image_features/gt_with_ibc.npy'
            'gt': os.path.join(gt_path, 'gt.npy'),
            # 'id to name map'
            'id2name': os.path.join(gt_path, 'id2name.json'),
            'cast_check': os.path.join(io_path, 'ca_lookup.json'),
            'res10ssd_prototxt_path': os.path.join(io_path, 'face_detection_ssd/deploy.prototxt'),
            'res10ssd_model_path': os.path.join(io_path, 'face_detection_ssd/res10_300x300_ssd_iter_140000.caffemodel'),
            'use_cuda': False
        })
        return params
    
    def set_config(self, config: dict) -> None:
        self.config = RuntimeConfig.from_dict(config)

    def get_config(self) -> dict:
        return asdict(self.config)

    def detect_batch(self, frames):
        """Args: frames
            Return:
                cropped_lst, bb_lst, index_lst"""

        def _crop_face(frame, b, h, w):
            x1, x2 = b[0], b[2]
            y1, y2 = b[1], b[3]
            return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        cropped_lst = []
        index_lst = []
        bb_lst = []
        t_s = time.time()

        if all([f.shape == frames[0].shape for f in frames]):
            boxes, probs, keypoints = self.detector.detect(
                [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], landmarks=True)
        else:
            boxes = []
            probs = []
            keypoints = []
            for f in frames:
                box, prob, keypoint = self.detector.detect([cv2.cvtColor(f, cv2.COLOR_BGR2RGB)],
                                                           landmarks=True)
                boxes.extend(box)
                probs.extend(prob)
                keypoints.extend(keypoint)

        res = []
        for b, p, k in zip(boxes, probs, keypoints):
            if p[0] is not None:
                res.append([{
                    'box': b[i],
                    'confidence': p[i],
                    'keypoints': {
                        'left_eye': k[i][0],
                        'right_eye': k[i][1],
                        'nose': k[i][2],
                        'mouth_left': k[i][3],
                        'mouth_right': k[i][4]
                    }
                } for i in range(len(p))])
            else:
                res.append(None)
        assert len(frames) == len(res)
        for i, f in enumerate(frames):
            if res[i] is None:
                continue
            h, w, _ = f.shape
            detections = [r for r in res[i] if r['confidence'] > 0.96]
            if len(detections) > 0:
                for det_ind, det in enumerate(detections):
        
                    b = [float(bb) for bb in det['box']]
                    b[0], b[2] = round(b[0]/w, 4), round(b[2]/w, 4)
                    b[1], b[3] = round(b[1]/h, 4), round(b[3]/h, 4)

                    if self._box_size(b) < self.config.min_box_size:
                        continue

                    bb_lst.append(b)

                    face = _crop_face(f, [int(round(bi))
                                      for bi in det['box']], h, w)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    cropped_lst.append(face)
                    index_lst.append(i)

        return cropped_lst, bb_lst, index_lst
    
    def _box_size(self, box: List[float]) -> float:
        return abs(box[2] - box[0]) * abs(box[3] - box[1])

    def _resize(self, image: np.ndarray, scale=1, max_size=3840):
        h, w = image.shape[:2]
        maxdim = max(h, w)
        if maxdim > max_size:
            scale = min(max_size/maxdim, scale)
        if scale < 1:
            h_new, w_new = int(h*scale), int(w*scale)
            return cv2.resize(image, (w_new, h_new))
        return
    
    
    def _tag_frames(self, frames, threshold_simi, threshold_simi_cast, threshold_cluster=0.3, cluster_ratio=0.1, cluster_flag=False, content_id=None, restrict_list: Optional[List[str]]=None):
        # get cast pool
        cast_pool = None
        if content_id:
            cast_pool = self.cast_check.get(content_id, None)
        elif restrict_list:
            cast_pool = restrict_list
        elif os.path.exists(os.path.join(self.pool_path, 'restrict.txt')):
            with open(os.path.join(self.pool_path, 'restrict.txt'), 'r') as f:
                cast_pool = [celeb.strip() for celeb in f.readlines()]
        
        if not self.logged:
            self.logged = True
            logger.debug(f"threshold_simi: {threshold_simi}, threshold_simi_cast: {threshold_simi_cast}, cluster_flag: {cluster_flag}")
            logger.info(f"Main cast pool: {cast_pool}")
            if cast_pool:
                cast_pool_invalid = [ c for c in cast_pool if len(self.name2id[c]) == 0 ]

                if cast_pool_invalid:
                    logger.warning("people in cast pool but not in ground truth: " + " ".join(cast_pool_invalid))

        # detect faces
        for i, f in enumerate(frames):
            maxdim = max(f.shape)
            if maxdim > 3840:
                frames[i] = self._resize(f)

        cropped_lst, bb_lst, index_lst = self.detect_batch(frames)
        logger.info(
            f"Content id {content_id}, Celeb: # images has faces: {len(set(index_lst))}, # faces detected {len(index_lst)}, total # images: {len(frames)}")
        if not cropped_lst:
            return defaultdict(list)
        cropped_lst_new = []
        for crop in cropped_lst:
            c = cv2.resize(crop, (112, 112))
            # transpose input to (3, h, w)
            c = np.transpose(c, (2, 0, 1))
            cropped_lst_new.append(c)
        cropped_lst = cropped_lst_new

        f1s = self.model.get_feature(np.array(cropped_lst))

        simi = np.dot(self.im_pool_feats, np.array(f1s).T)
        top_idx = np.argmax(simi, 0)
        scores = [float(simi[idx][i]) for i, idx in enumerate(top_idx)]

        # create a intermediate result list to store all original tags
        res_inter = defaultdict(list)
        res_inter_tmp = defaultdict(list)

        for idx, (score, topk, bbox, ind) in enumerate(zip(scores, top_idx, bb_lst, index_lst)):
            if self.gt[topk] in self.id2name:
                res_inter_tmp[idx] = (self.id2name[self.gt[topk]], score)
            else:
                res_inter_tmp[idx] = ("", score)

            if self.gt[topk] not in self.id2name:
                continue
            
            if cast_pool is not None and self.id2name[self.gt[topk]] in cast_pool:
                target_thresh = threshold_simi_cast
            else:
                target_thresh = threshold_simi

            if score >= target_thresh:
                res_inter[idx] = (self.id2name[self.gt[topk]], score, list(bbox),
                                    frames[ind].shape[0], frames[ind].shape[1])

        tmp = {index_lst[k]: (v[0], v[1])
                for k, v in res_inter_tmp.items()}  # if v[1]>0.4}
        logger.info(f"Raw predictions: {tmp}")
        # create a dictionary to store the mapping of name and face index
        name_fraid = defaultdict(set)
        for idx, v in res_inter.items():
            name_fraid[v[0]].add(idx)
        logger.info(f"Name mapping: {name_fraid}")

        # assign all clusters the name tagged

        if cluster_flag:
            # cluster faces
            face_im_simi = np.dot(np.array(f1s), np.array(f1s).T)
            clusters = clustering(face_im_simi, threshold_cluster)

            #n_clusters = len([k for k,v in name_fraid.items()])
            #clusters = km(np.array(f1s), n_clusters)

            logger.info(f"cluster sets: {clusters}")
            logger.info(
                f"clusters: {[[res_inter_tmp[i][0] for i in s] for s in clusters]}")
            # majority vote to decide if adapt the current cluster
            ids_name = {}
            for c in clusters:
                max_inter = 0
                for k, v in name_fraid.items():
                    if len(c.intersection(v)) > max_inter and len(c.intersection(v))/len(c) > cluster_ratio:
                        side_nodes_scores = [res_inter[idx][1]
                                                for idx in c.intersection(v)]
                        mean_score = np.mean(side_nodes_scores)
                        for i in c:
                            if i in v:
                                ids_name[i] = (k, None, 'main', index_lst[i])
                            else:
                                ids_name[i] = (
                                    k, mean_score, "cluster", index_lst[i])
                        max_inter = len(c.intersection(v))

            res_inter = ids_name

        res = defaultdict(list)
        for k, v in res_inter.items():
            ind = index_lst[k]
            score = scores[k] if v[1] is None else v[1]
            bbox = bb_lst[k]
            topk = top_idx[k]
            res[index_lst[k]].append((
                v[0], score, list(
                    bbox), frames[ind].shape[0], frames[ind].shape[1]
            ))
        logger.info(f"Content id {content_id}, Celeb prediction: {res}")
        return res
    
    def tag(self, img: np.ndarray) -> List[FrameTag]:
        content_id = self.config.content_id
        res = self._tag_frames([img], self.config.thres, self.config.thres_cast, content_id=content_id, restrict_list = self.config.restrict_list)
        if len(res[0]) == 0:
            res = []
        else:
            res = res[0]
        return [FrameTag.from_dict({"text": text, "confidence": conf, "box": {"x1": round(box[0], 4), "y1": round(box[1], 4), "x2": round(box[2], 4), "y2":  round(box[3], 4)}}) for text, conf, box, _, _ in res]
    
def clustering(simi_matrix, thre):
    cluster = []
    n = len(simi_matrix)
    m = len(simi_matrix[0])
    for i in range(n):
        for j in range(i+1, m):
            if simi_matrix[i][j] > thre:
                cluster.append((i, j))
    G = nx.Graph()
    G.add_edges_from(cluster)
    new_cluster = list(nx.connected_components(G))
    return new_cluster

def km(x, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=22)
    kmeans.fit(x)
    logger.info(f"**** count {len(kmeans.labels_)} ---- {kmeans.labels_}")
    ret = pd.Series(range(len(kmeans.labels_))).groupby(
        kmeans.labels_, sort=False).apply(list).tolist()
    return [set(l) for l in ret]