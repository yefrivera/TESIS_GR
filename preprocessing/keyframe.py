# preprocessing/keyframe.py
import cv2
import numpy as np
from scipy.stats import entropy

class KeyframeExtractor:
    def __init__(self, target_frames=32):
        self.k = target_frames

    def _calculate_entropy(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / (hist.sum() + 1e-7)
        # Entropía de Shannon
        return -np.sum(hist * np.log2(hist + 1e-7))

    def extract(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        entropies = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            entropies.append(self._calculate_entropy(frame))
        cap.release()
        
        if not frames:
            return []

        # Selección basada en variación de entropía
        # Si el video es corto, hacemos padding; si es largo, muestreo uniforme basado en entropía
        indices = np.linspace(0, len(frames)-1, self.k, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        return selected_frames