import h5py
import torch
import clip
import argparse
import numpy as np
import pandas as pd
import cv2
import json

from torch._C import device
from configs.config import init_config

from utils.extractor_model import ExtractorModule


# def get_arg():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--query', type=str)
#     args = parser.parse_args()
#     return args

my_cfg = init_config()

if __name__ == '__main__':

    devices = my_cfg["device"]["device"]
    model_name = my_cfg["CLIP"]["model_name"]
    model = ExtractorModule(model_name=model_name, device=devices)

    hf = h5py.File('keyframes.h5', 'r')

    ids = hf.get('ids')
    ids = np.array(ids)
    for i in range(len(ids)):
        ids[i] = ids[i].decode('utf-8')
    
    features = hf.get('features')
    features = np.array(features)

    with open('keyframes_id.json', 'r') as f:
        keyframes_id_path = json.load(f)
    

    query = input("Query: ")

    best_frame_idx = model.search_frame(search_query=query, frame_features=features,
                                        display_results_count=4)
    
    
    image1 = cv2.imread(keyframes_id_path[ids[best_frame_idx[0]][0]])
    image2 = cv2.imread(keyframes_id_path[ids[best_frame_idx[1]][0]])
    image3 = cv2.imread(keyframes_id_path[ids[best_frame_idx[2]][0]])
    image4 = cv2.imread(keyframes_id_path[ids[best_frame_idx[3]][0]])

    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)
    cv2.imshow('image3', image3)
    cv2.imshow('image4', image4)

    cv2.waitKey(0)
    