import torch
import os
import argparse
import tqdm
import pandas as pd

from utils.model import ExtractorModule

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframe_folder_path', type=str)
    parser.add_argument('--cuda', action='store_true', default=False) 
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_arg()

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    print("Load extractor")
    extractor = ExtractorModule(device=device)
    print("Using", device.type)

    keyframe_folder_path = args.key_frame_folder_path
    shot_folder_list = os.listdir(keyframe_folder_path)
    shot_folder_list = sorted(shot_folder_list)

    keyframe_id = []
    keyframes_features = torch.empty([0, 512], dtype=torch.float16).to(device)

    for folder in tqdm(shot_folder_list):
        folder_path = os.path.join(keyframe_folder_path, folder)
        ids = os.listdir(folder_path)
        features = extractor.ExtractKeyframes(folder_keyframe_path=folder_path)

        keyframes_features = torch.cat((keyframes_features, features))
        keyframe_id += ids
    
    keyframes_features = keyframes_features.tolist()

    data = pd.DataFrame(zip(keyframe_id, keyframes_features), columns=['id', 'features'])
    data.to_csv('keyframes_features.csv')


    
