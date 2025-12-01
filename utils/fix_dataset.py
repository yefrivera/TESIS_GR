import os
import glob

def sanitize_dataset(root_dir):
    """
    1. Renombra archivos para quitar espacios y paréntesis.
    2. Genera nuevos archivos de lista (train_list.txt).
    """
    print(f"Procesando directorio: {root_dir}")
    
    # Extensiones de video soportadas
    extensions = ['*.mp4', '*.avi', '*.mov']
    
    # Clases encontradas (Carpetas)
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    print(f"Clases detectadas: {classes}")
    
    new_index_lines = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(root_dir, class_name)
        
        # Obtener todos los videos
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(class_path, ext)))
            
        print(f"  > Clase '{class_name}': {len(files)} videos encontrados.")
        
        for i, file_path in enumerate(files):
            directory = os.path.dirname(file_path)
            extension = os.path.splitext(file_path)[1]
            
            # NUEVO NOMBRE SEGURO: clase_indice.mp4 (ej. agarrar_001.mp4)
            new_name = f"{class_name}_{i+1:03d}{extension}"
            new_path = os.path.join(directory, new_name)
            
            # Renombrar archivo físico si es necesario
            if file_path != new_path:
                try:
                    os.rename(file_path, new_path)
                except OSError as e:
                    print(f"Error renombrando {file_path}: {e}")
                    new_path = file_path # Mantener nombre anterior si falla
            
            # Guardar ruta normalizada (reemplazar \ por / para compatibilidad total)
            # Usamos ruta relativa segura
            relative_path = os.path.relpath(new_path, start=".")
            relative_path = relative_path.replace("\\", "/")
            
            new_index_lines.append(f"{relative_path} {class_idx}\n")

    return new_index_lines

if __name__ == "__main__":
    # Ajusta estas rutas a donde tengas tus carpetas REALES
    train_root = "./data/raw/train" 
    val_root = "./data/raw/val"
    
    # 1. Sanear Entrenamiento
    if os.path.exists(train_root):
        lines = sanitize_dataset(train_root)
        with open("./data/train_list.txt", "w") as f:
            f.writelines(lines)
        print("✅ train_list.txt regenerado correctamente.")
    else:
        print(f"❌ Error: No encuentro la carpeta {train_root}")

    # 2. Sanear Validación
    if os.path.exists(val_root):
        lines = sanitize_dataset(val_root)
        with open("./data/val_list.txt", "w") as f:
            f.writelines(lines)
        print("✅ val_list.txt regenerado correctamente.")
    else:
        print(f"❌ Error: No encuentro la carpeta {val_root}")