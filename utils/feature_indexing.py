import faiss
import os
import h5py
import numpy as np
import pandas as pd
import clip
import torch
import json
from tqdm import tqdm
import argparse

def load_data_h5(path: str) -> dict:
    hf = h5py.File(path, 'r')
    data = {'feature': np.array(hf.get('features')),
            'keyframe_id': np.array(hf.get('ids'))}
    print(f'(+) Loaded {data["feature"].shape} features of {os.path.basename(path)}')
    hf.close()
    return data

def save_img_urls(keyframe_ids: list, path_img_url: str):
    keyframe_ids = [keyframe_id.decode("utf-8") for keyframe_id in keyframe_ids]
    keyframe_ids = [keyframe_id.split(".")[0] for keyframe_id in keyframe_ids]
    img_urls = []
    for keyframe_id in keyframe_ids:
        video_id = keyframe_id.split("_")[0]
        subset = 'V3C1' if int(video_id) < 7476 else 'V3C2'
        img_urls.append(os.path.join(subset, video_id, f'{keyframe_id}.png'))

    df = pd.DataFrame({'keyframe_id': keyframe_ids,
                       'url': img_urls,
                       'index': np.arange(0, len(img_urls)).tolist()})
    df.to_csv(path_img_url)
    print(f'(+) Image url file is save at {path_img_url}')


def faiss_index(feature_path, save_path, dataframe_path, dims=512):
    print(f'(+) Indexing')
    index = faiss.IndexFlatL2(dims)

    keyframe_ids = []
    data = load_data_h5(feature_path)

    for feature in tqdm(data['feature']):
        index.add(feature.reshape(1, dims).astype(np.float32))
    faiss.write_index(index, save_path)
    print('\n')
    print(f'(+) Index file is save at {save_path}')
    save_img_urls(keyframe_ids=keyframe_ids, path_img_url=dataframe_path)

def load_faiss_index(faiss_index_path, image_dataframe_path):
    print(f'(+) Loading index')
    index = faiss.read_index(faiss_index_path)
    df_image = pd.read_csv(image_dataframe_path)
    return index, df_image

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indexed_features_path', type=str)
    parser.add_argument('--dataframe_index_path', type=str)
    parser.add_argument('--mapping_kf2shot', type=str)

    args = parser.parse_args()
    return args

class IndexingRetrievalModel:

    def __init__(self, args, my_cfg) -> None:

        n_gpus = faiss.get_num_gpus()
        print("number of GPUs:", n_gpus)

        cpu_index = faiss.read_index(args.indexed_features_path)
        self.indexed_feature =faiss.index_cpu_to_all_gpus(cpu_index)

        self.df_image = pd.read_csv(args.dataframe_index_path)
        self.df_image['url'] = '/mlcv/Databases/VBS/Processed_Data/Thumbnail/TransNetV2_200x113/' + self.df_image['url']
        keyframe_id = self.df_image['keyframe_id']
        video_id = [id.split('_')[0] for id in keyframe_id]
        df_vid = pd.DataFrame({'video_id': video_id})
        self.df_image = pd.concat([self.df_image, df_vid], axis=1)

        content = open(args.mapping_kf2shot)
        self.mapping = json.load(content)

        self.model, self.preprocess = clip.load(my_cfg["CLIP"]["model_name"])

    def clip_extract_feature(self, text):
        text_tokens = clip.tokenize(text).cuda()
        with torch.no_grad():
            return self.model.encode_text(text_tokens).float().cpu().numpy()
    
    def retrieval(self, text_query):
        text_feature = self.clip_extract_feature(text_query)
        f_dists, f_ids = self.indexed_feature.search(text_feature, k=3000)

        df_res = pd.DataFrame({'index': list(np.array(f_ids).flat),
                               'dist': list(np.array(f_dists).flat)})
        
        df_res = pd.merge(df_res, self.df_image, how='left', on='index')
        result_imgs = [{
                'dataset': row['url'].split('/')[0],
                'video_id': row['video_id'],
                'shot_id': self.mapping.get(row['keyframe_id']),
                'frame_id': row['keyframe_id'],
                # 'keyframe_name': row['url'].split('/')[-1],
                'thumbnail_path': row['url'],
                'score' : row['dist']
                } for i, row in df_res.iterrows()] 
        
        return result_imgs
