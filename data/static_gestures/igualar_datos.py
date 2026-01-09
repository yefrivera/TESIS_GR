import pandas as pd

archivo_entrada = 'static_gestures.csv'
archivo_salida = 'static_gestures_v2.csv'

print(f"Leyendo {archivo_entrada}...")

try:
    # 1. Cargar el archivo ignorando las líneas defectuosas que daban error antes
    # engine='python' y on_bad_lines='skip' ayudan a saltar las filas con columnas extra
    df = pd.read_csv(archivo_entrada, header=None, on_bad_lines='skip', engine='python')

    # 2. Calcular cuál es la cantidad mínima de muestras que tiene un gesto
    conteo_por_clase = df[0].value_counts()
    min_muestras = conteo_por_clase.min()
    
    print("\n--- Estado Actual ---")
    print(conteo_por_clase)
    print(f"\nSe recortarán todos los gestos a: {min_muestras} muestras cada uno.")

    # 3. Equilibrar: Tomar aleatoriamente 'min_muestras' de cada grupo
    # random_state=42 asegura que siempre elija las mismas filas si lo corres de nuevo
    df_balanced = df.groupby(0).apply(lambda x: x.sample(n=min_muestras, random_state=42))

    # Limpiar el índice extra que crea el groupby
    df_balanced = df_balanced.reset_index(drop=True)

    # 4. Guardar en un archivo nuevo
    df_balanced.to_csv(archivo_salida, index=False, header=False)

    print(f"\n¡Éxito! Archivo guardado como: {archivo_salida}")
    print("--- Nuevo Conteo ---")
    print(df_balanced[0].value_counts())

except Exception as e:
    print(f"Ocurrió un error: {e}")