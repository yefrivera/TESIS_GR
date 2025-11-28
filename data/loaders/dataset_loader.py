# data/loaders/dataset_loader.py
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from preprocessing.keyframe import KeyframeExtractor
from preprocessing.denoising import WaveletDenoiser
from models.detection import HandDetector

class GestureDataset(Dataset):
    def __init__(self, video_paths, labels, config, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        
        # Inicializar procesadores
        self.keyframe_extractor = KeyframeExtractor(target_frames=config.SAMPLE_DURATION)
        self.denoiser = WaveletDenoiser(wavelet=config.WAVELET_TYPE)
        self.detector = HandDetector()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        # 1. Extraer Keyframes (Temporal)
        frames = self.keyframe_extractor.extract(path)
        
        processed_clip = []
        for frame in frames:
            # 2. Detección de Mano (Espacial)
            hand_roi = self.detector.crop_hand(frame)
            
            # 3. Denoising
            clean_frame = self.denoiser.apply(hand_roi)
            
            # 4. Resize y Normalización
            clean_frame = cv2.resize(clean_frame, (self.config.IMG_SIZE, self.config.IMG_SIZE))
            processed_clip.append(clean_frame)
            
        # Convertir a Tensor: (Time, H, W, C) -> (C, Time, H, W)
        clip = np.array(processed_clip, dtype=np.float32)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2) # C, D, H, W
        clip = clip / 255.0 # Normalización simple
        
        return clip, label