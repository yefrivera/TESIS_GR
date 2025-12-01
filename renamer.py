# utils/renamer.py
import os

def rename_dataset_files(root_dir):
    """
    Recorre las carpetas train/ y val/ y renombra todos los videos
    a un formato estándar: Clase_001.mp4, Clase_002.mp4, etc.
    Elimina espacios, paréntesis y caracteres raros.
    """
    print(f"--- Iniciando Renombrado en: {root_dir} ---")
    
    if not os.path.exists(root_dir):
        print(f"ERROR: No encuentro la ruta {root_dir}")
        return

    # Recorrer carpetas train y val
    for subset in ['train', 'val']:
        subset_path = os.path.join(root_dir, subset)
        
        if not os.path.exists(subset_path):
            print(f"Saltando {subset} (no existe)...")
            continue
            
        print(f"\nProcesando carpeta: {subset}...")
        
        # Obtener lista de clases (carpetas de gestos)
        classes = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        
        for class_name in classes:
            class_dir = os.path.join(subset_path, class_name)
            files = os.listdir(class_dir)
            
            # Filtrar solo videos
            video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            video_files.sort() # Ordenar para mantener consistencia
            
            count = 0
            for index, filename in enumerate(video_files):
                # Obtener extensión (.mp4)
                _, ext = os.path.splitext(filename)
                
                # Crear NUEVO nombre limpio: Clase_001.mp4
                # Usamos zfill(3) para tener 001, 002... (mejor ordenamiento)
                new_name = f"{class_name}_{str(index + 1).zfill(3)}{ext}"
                
                old_path = os.path.join(class_dir, filename)
                new_path = os.path.join(class_dir, new_name)
                
                # Renombrar solo si el nombre es diferente
                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        count += 1
                    except OSError as e:
                        print(f"Error renombrando {filename}: {e}")
            
            print(f"  > Clase '{class_name}': {count} videos renombrados.")

    print("\n✅ Proceso completado. Tus videos ahora tienen nombres limpios.")

if __name__ == "__main__":
    # Asegúrate de que esta ruta apunte a tu carpeta 'custom_dataset'
    # Según tu estructura anterior, debería ser esta:
    target_folder = "./data/raw"
    
    rename_dataset_files(target_folder)