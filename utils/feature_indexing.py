import faiss
import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    