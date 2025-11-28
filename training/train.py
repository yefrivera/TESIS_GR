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
    """Lee los archivos de texto generados por el script de mapeo."""
    video_paths = []
    labels = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"No se encontró el archivo de índice: {txt_path}")
        
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue # Saltar líneas vacías
            
            # CORRECCIÓN: Usar rsplit con maxsplit=1
            # Separa por el ÚLTIMO espacio encontrado.
            # Ejemplo: "./data/mi video (1).mp4 2" -> ["./data/mi video (1).mp4", "2"]
            parts = line.rsplit(' ', 1)
            
            if len(parts) == 2:
                video_paths.append(parts[0])
                try:
                    labels.append(int(parts[1]))
                except ValueError:
                    print(f"Advertencia: No se pudo procesar la etiqueta en la línea: {line}")
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
        print(f"Error cargando listas: {e}")
        return

    print(f"Datos de entrenamiento: {len(train_paths)} videos")
    print(f"Datos de validación: {len(val_paths)} videos")

    # Instanciar Datasets y Loaders
    # num_workers > 0 paraleliza la carga (Keyframe extraction + Denoising)
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
    
    # Optimizador Adam, alineado con la configuración experimental de Xi et al. [cite: 2110]
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # Scheduler: Decaimiento exponencial del learning rate para refinar la convergencia [cite: 2113]
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0

    # 5. Bucle de Entrenamiento
    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Barra de progreso para el epoch actual
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}]")
        
        for inputs, labels in loop:
            # Mover tensores a GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Actualizar Learning Rate
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f"\nResumen Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # 6. Validación y Guardado
        print("Iniciando validación...")
        val_acc, val_f1, _ = evaluate_model(model, val_loader, device)
        
        # Guardar el mejor modelo (Checkpoint)
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("weights", "best_model_3d_resnext.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Nuevo mejor modelo guardado (Acc: {val_acc:.4f}) en {save_path}")

    print("Entrenamiento completado.")

if __name__ == "__main__":
    train_model()