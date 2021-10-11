import clip
import numpy as np
import torch
import cv2
import os
from PIL import Image
import math
from tqdm import tqdm

import argparse
import h5py

from configs.config import init_config

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path', type=str)

    args = parser.parse_args()
    return args

class RetrievalModel:
    def __init__(self, args, my_cfg) -> None:
        
        self.data = h5py.File(args.features_path, 'r')
        self.model_name = my_cfg["CLIP"]["model_name"]
        self.device = my_cfg["device"]["device"]

        self.features = self.data.get('features')
        self.features = np.array(self.features)
        # self.features = torch.from_numpy(self.features).to(self.device)
        # self.features = self.features.to(torch.float16)

        self.count = len(self.features)

        self.ids = self.data.get('ids')
        self.ids = np.array(self.ids)
        for i in range(len(self.ids)):
            self.ids[i] = self.ids[i].decode('utf-8')

        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    def retrieval(self, query):
        
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(query).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # print(self.features.shape)
        text_features = text_features.to("cpu").numpy()
        
        similarities = (100 * self.features @ text_features.T)
        best_frame_idx = similarities.argsort(axis=0)[:][::-1]
        results = []

        for i in best_frame_idx:
            results.append(self.ids[i][0])
        
        return results
