# main.py
import cv2
import torch
import numpy as np
from collections import deque
import time

# Importar módulos locales
from data.raw.val.confirmar.config import Config
from models.detection import HandDetector
from preprocessing.denoising import WaveletDenoiser
from models.temporal_models import ResNeXt101_3D_CBAM

def main():
    # 1. Inicialización
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    print(f"Iniciando sistema en: {device}")

    # Cargar Modelo Entrenado (Simulado)
    print("Cargando modelo 3D-ResNeXt + CBAM...")
    model = ResNeXt101_3D_CBAM(
        num_classes=cfg.NUM_CLASSES, 
        sample_duration=cfg.SAMPLE_DURATION
    ).to(device)
    
    # Aquí cargarías los pesos reales:
    # model.load_state_dict(torch.load("weights/best_model.pth"))
    model.eval()

    # Herramientas de Preprocesamiento
    detector = HandDetector(detection_con=0.8)
    denoiser = WaveletDenoiser(wavelet=cfg.WAVELET_TYPE) # [cite: 1460]

    # Buffer temporal para gestos dinámicos (Ventana deslizante)
    # HaGRIDv2 sugiere una cola de los últimos N frames [cite: 2535]
    frame_buffer = deque(maxlen=cfg.SAMPLE_DURATION)
    
    # 2. Captura de Video
    cap = cv2.VideoCapture(0) # Webcam 0
    
    # Clases (Ejemplo Jester/HaGRID)
    classes = ['Afirmar', 'Bajar La Mano', 'Confirmar', 'Levangtar La Mano', 'Pellizco'] 

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # A. Detección de Mano (ROI)
        # Aislar la mano reduce ruido de fondo [cite: 2499]
        hand_roi = detector.crop_hand(frame)
        
        # B. Preprocesamiento (Denoising)
        # Aplicar Wavelet Thresholding [cite: 1462]
        clean_frame = denoiser.apply(hand_roi)
        
        # C. Preparación para Tensor
        # Resize y Normalización
        input_frame = cv2.resize(clean_frame, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # Añadir al buffer temporal
        frame_buffer.append(input_frame)

        # D. Inferencia (Solo si el buffer está lleno)
        if len(frame_buffer) == cfg.SAMPLE_DURATION:
            # Convertir buffer a Tensor: (1, C, D, H, W)
            # Stack frames -> (D, H, W, C)
            clip = np.array(frame_buffer, dtype=np.float32)
            # Permutar -> (1, C, D, H, W) para PyTorch 3D Conv
            clip_tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
            clip_tensor = (clip_tensor / 255.0).to(device)

            with torch.no_grad():
                outputs = model(clip_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_idx].item()

            # Mostrar resultado si la confianza es alta
            if confidence > 0.7:
                label = f"{classes[pred_idx]} ({confidence:.2f})"
                cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)
                
                # Visualización en consola para depuración
                print(f"Gesto detectado: {label}")

        # Visualización
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1)
        
        cv2.imshow('Gesture Recognition Thesis Demo', frame)
        # Mostrar también lo que "ve" la red (ROI denoisada)
        cv2.imshow('Network Input (ROI)', clean_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    