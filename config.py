# config.py
import torch

class Config:
    # --- GESTIÓN DE DATOS ---
    # Rutas para el dataset personalizado (según el script de mapeo previo)
    DATASET_ROOT = "./data/raw"
    TRAIN_LIST = "./data/train_list.txt"
    VAL_LIST = "./data/val_list.txt"
    
    # --- PREPROCESAMIENTO ---
    # Se define en 32 frames porque ofrece el mejor balance entre precisión y 
    # costo computacional (GFLOPs) según los experimentos de Xi et al..
    SAMPLE_DURATION = 32       
    
    # Reducción a 112x112 píxeles para mantener la eficiencia del modelo 3D 
    # sin perder características espaciales críticas[cite: 1337].
    IMG_SIZE = 112             
    
    # Uso de Daubechies 6 (db6) para eliminación de ruido, ya que preserva mejor 
    # los bordes de la mano que los filtros gaussianos[cite: 284].
    WAVELET_TYPE = 'db6'       
    
    # Tamaño del buffer para calcular la entropía de Shannon y seleccionar keyframes[cite: 251].
    ENTROPY_BUFFER = 64        
    
    # --- ARQUITECTURA DEL MODELO (3D-ResNeXt + CBAM) ---
    # Configurado para 4 clases iniciales (ej. Swipe, Zoom, Click, NoGesture).
    # Nota: HaGRIDv2 original tiene muchas más, pero se reduce para el prototipo específico.
    NUM_CLASSES = 4            
    
    # Cardinalidad estándar para ResNeXt, controlando el número de grupos de convolución[cite: 1199].
    CARDINALITY = 32           
    
    # Profundidad temporal para el módulo de atención 3D-CBAM[cite: 1236].
    DEPTH_DIM = 32             
    
    # --- HIPERPARÁMETROS DE ENTRENAMIENTO ---
    # Tamaño de lote ajustado para GPUs estándar (ej. 8GB/16GB VRAM).
    BATCH_SIZE = 8             
    
    # Tasa de aprendizaje inicial para optimizador Adam[cite: 388].
    LEARNING_RATE = 0.001      
    
    # El modelo tiende a converger cerca de la época 20, por lo que 50 es un margen seguro[cite: 1375].
    EPOCHS = 50
    
    OPTIMIZER = 'adam'         
    
    # Detección automática de hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'