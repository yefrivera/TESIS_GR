import os

nombre_archivo = 'static_gestures.csv'
nombre_temporal = 'static_gestures_temp.csv'

print(f"Procesando {nombre_archivo}...")

try:
    with open(nombre_archivo, 'r') as entrada, open(nombre_temporal, 'w') as salida:
        for linea in entrada:
            # Verificamos si la línea empieza exactamente con "5,"
            # Esto evita cambiar un "50," o un "15," por error.
            if linea.startswith('5,'):
                # Reemplazamos el primer '5' por '4' y guardamos el resto de la línea igual
                nueva_linea = '4' + linea[1:]
                salida.write(nueva_linea)
            else:
                # Si no empieza con 5, la dejamos tal cual
                salida.write(linea)

    # Reemplazar el archivo original con el modificado
    os.remove(nombre_archivo)
    os.rename(nombre_temporal, nombre_archivo)
    
    print("¡Listo! Se han cambiado los índices 5 por 4 correctamente.")

except FileNotFoundError:
    print(f"Error: No se encontró el archivo {nombre_archivo}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")