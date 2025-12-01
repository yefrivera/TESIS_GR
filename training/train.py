# training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm  # Barra de progreso para visualizar entrenamiento

# Importaciones locales
from config import Config
from models.temporal_models import ResNeXt101_3D_CBAM
from data.loaders.dataset_loader import GestureDataset
from training.evaluate import evaluate_model  # Asegúrate de tener este script creado
from torch.utils.data import DataLoader

def load_data_from_txt(txt_path):
    """
    Lee los archivos de texto y valida las rutas.
    Maneja el formato: 'ruta/al/video.mp4 etiqueta'
    """
    video_paths = []
    labels = []
    
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"No se encontró el archivo de índice: {txt_path}")
        
    print(f"Leyendo índice: {txt_path}...")
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue # Saltar líneas vacías
            
            # Usar rsplit para separar solo el último espacio (la etiqueta)
            # Esto protege si el nombre del archivo tuviera espacios (aunque ya los limpiaste)
            parts = line.rsplit(' ', 1)
            
            if len(parts) == 2:
                raw_path = parts[0].strip()
                label_str = parts[1].strip()
                
                # Normalizar ruta para Windows/Linux (convierte / a \ automáticamente si es necesario)
                clean_path = os.path.normpath(raw_path)
                
                # Verificación de seguridad: ¿Existe el archivo?
                if not os.path.exists(clean_path):
                    # Intenta buscarlo relativo a la raíz si falla la ruta absoluta/relativa dada
                    if os.path.exists(os.path.join('.', clean_path)):
                        clean_path = os.path.join('.', clean_path)
                    else:
                        print(f"⚠️ Advertencia: Video no encontrado en disco: {clean_path}")
                        continue 

                try:
                    video_paths.append(clean_path)
                    labels.append(int(label_str))
                except ValueError:
                    print(f"❌ Error: No se pudo procesar la etiqueta numérica en la línea: {line}")
    
    return video_paths, labels

def train_model():
    # 1. Configuración Inicial
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    print(f"--- Iniciando configuración de entrenamiento en {device} ---")
    
    # Crear directorio para guardar pesos si no existe
    os.makedirs("weights", exist_ok=True)

    # 2. Carga de Datos (Dataset Real)
    print("Cargando índices de dataset...")
    try:
        train_paths, train_labels = load_data_from_txt(cfg.TRAIN_LIST)
        val_paths, val_labels = load_data_from_txt(cfg.VAL_LIST)
    except Exception as e:
        print(f"Error fatal cargando listas: {e}")
        return

    if len(train_paths) == 0:
        print("❌ Error: No se cargaron videos de entrenamiento. Revisa train_list.txt")
        return

    print(f"✅ Datos de entrenamiento cargados: {len(train_paths)} videos")
    print(f"✅ Datos de validación cargados: {len(val_paths)} videos")

    # Instanciar Datasets y Loaders
    # num_workers=0 es CRÍTICO para Windows + MediaPipe
    train_dataset = GestureDataset(train_paths, train_labels, cfg)
    val_dataset = GestureDataset(val_paths, val_labels, cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Inicialización del Modelo
    print("Construyendo modelo 3D-ResNeXt-101 + 3D-CBAM...")
    model = ResNeXt101_3D_CBAM(
        num_classes=cfg.NUM_CLASSES, 
        sample_duration=cfg.SAMPLE_DURATION
    ).to(device)

    # 4. Configuración de Optimización
    criterion = nn.CrossEntropyLoss()
    
    # [cite_start]Optimizador Adam [cite: 2110]
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # [cite_start]Scheduler: Decaimiento exponencial [cite: 2113]
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0

    # 5. Bucle de Entrenamiento
    print(f"Iniciando ciclo de {cfg.EPOCHS} épocas...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Barra de progreso
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        
        for inputs, labels in loop:
            # Mover a GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Métricas batch
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Step del Scheduler
        scheduler.step()
        
        # Métricas de época
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f"\nResumen Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # 6. Validación
        print("Validando...")
        val_acc, val_f1, _ = evaluate_model(model, val_loader, device)
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("weights", "model_3d_resnext.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Nuevo modelo guardado (Acc: {val_acc:.4f}) en {save_path}")

    print("Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    train_model()