# models/detection.py
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode='mediapipe', detection_con=0.7):
        self.mode = mode
        if self.mode == 'mediapipe':
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=detection_con
            )
        # Aquí agregarías la carga de YOLOv10-HaGRIDv2 en el futuro

    def crop_hand(self, frame):
        """Retorna el ROI de la mano o el frame original si no detecta nada"""
        h, w, c = frame.shape
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            # Lógica simple para una mano (se puede expandir a 2)
            hand_landmarks = results.multi_hand_landmarks[0]
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            
            xmin, xmax = int(min(x_list) * w), int(max(x_list) * w)
            ymin, ymax = int(min(y_list) * h), int(max(y_list) * h)
            
            # Agregar padding
            pad = 20
            xmin, xmax = max(0, xmin-pad), min(w, xmax+pad)
            ymin, ymax = max(0, ymin-pad), min(h, ymax+pad)
            
            return frame[ymin:ymax, xmin:xmax]
            
        return frame # Retorna frame completo si no hay detección (fallback)