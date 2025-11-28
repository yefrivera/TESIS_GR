# preprocessing/denoising.py
import numpy as np
import pywt
import cv2

class WaveletDenoiser:
    def __init__(self, wavelet='db6', level=2):
        self.wavelet = wavelet
        self.level = level

    def apply(self, frame):
        """
        Aplica umbralización de ondícula modificada para preservar bordes de la mano.
        """
        # Normalizar a float 0-1
        img_float = frame.astype(float) / 255.0
        channels = []
        
        # Procesar por canal independientemente
        for i in range(3):
            coeffs = pywt.wavedec2(img_float[:,:,i], self.wavelet, level=self.level)
            # Estimación de umbral VisuShrink simplificada
            sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(frame.shape[0] * frame.shape[1]))
            
            # Umbralización suave
            new_coeffs = list(coeffs)
            for j in range(1, len(coeffs)):
                new_coeffs[j] = tuple([pywt.threshold(c, threshold, mode='soft') for c in coeffs[j]])
            
            channels.append(pywt.waverec2(new_coeffs, self.wavelet))
            
        denoised = cv2.merge(channels)
        return np.uint8(np.clip(denoised * 255, 0, 255))