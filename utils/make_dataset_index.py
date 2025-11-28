# utils/make_dataset_index.py
import os

def create_index(dataset_path, output_txt):
    # Obtener nombres de clases (orden alfabético para consistencia)
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Clases encontradas: {class_to_idx}")
    
    with open(output_txt, 'w') as f:
        for cls_name in classes:
            cls_path = os.path.join(dataset_path, cls_name)
            for video_name in os.listdir(cls_path):
                if video_name.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(cls_path, video_name)
                    # Escribir: ruta_del_video etiqueta_numerica
                    f.write(f"{video_path} {class_to_idx[cls_name]}\n")
    
    print(f"Índice guardado en: {output_txt}")

if __name__ == "__main__":
    # Generar para entrenamiento
    create_index("./data/raw/train", "./data/train_list.txt")
    # Generar para validación
    create_index("./data/raw/val", "./data/val_list.txt")