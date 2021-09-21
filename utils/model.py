import clip
import numpy as np
import torch
import cv2
import os
from PIL import Image
import math
import tqdm

class ExtractorModule:

    def __init__(self, model_name='ViT-B/32', device='cuda'):
        
        self.model_name = model_name
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def ExtractKeyframes(self, folder_keyframe_path, 
                          batch_size=256):
        
        file_names = [f for f in os.listdir(folder_keyframe_path) 
                      if os.path.isfile(os.path.join(folder_keyframe_path, f)) 
                      and f.endswith('.jpg') or f.endswith('.png')]
        
        for i in range(len(file_names)):
            file_names[i] = os.path.join(folder_keyframe_path, file_names[i])

        keyframes = []

        for path in file_names:
            # print(path)
            keyframe = cv2.imread(path)
            keyframes.append(Image.fromarray(keyframe[:, :, ::-1]))
        
        batches = math.ceil(len(keyframes) / batch_size)

        keyframe_features = torch.empty([0, 512], dtype=torch.float16).to(self.device)

        for i in tqdm(range(batches)):
            # print(f"Processing batch {i+1}/{batches}")

            batch_frames = keyframes[i * batch_size : (i + 1) * batch_size]
            batch_preprocessed = torch.stack([self.preprocess(keyframe) for keyframe in batch_frames]).to(self.device)
        
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_preprocessed)
                batch_features /= batch_features.norm(dim=-1, keepdim = True)
        
            keyframe_features = torch.cat((keyframe_features, batch_features))


        # print("Extract Features == Finished")
        return keyframe_features

    def ExtractText(self, query):
        
        with torch.no_grad():
            text_features = self.model.encode_text(clip.tokenize(query).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features


