# Implementación Avanzada: Reconocimiento Integral de Gestos Dinámicos

La implementación actual del reconocimiento de gestos dinámicos es eficiente pero limitada: solo analiza el movimiento cuando la mano adopta una pose específica de "Puntero" (dedo índice extendido). Esto impide reconocer gestos más complejos y naturales que involucran el movimiento de toda la mano, como un saludo.

Este documento detalla el plan para desarrollar un sistema más robusto que analice la secuencia de los **21 puntos clave de la mano** a lo largo del tiempo, permitiendo el reconocimiento simultáneo de la pose estática y del movimiento dinámico.

---

### Plan de Implementación en 3 Pasos

#### Paso 1: Modificar la Captura de Datos (`app.py`)

El objetivo principal es registrar y guardar el historial de movimiento de todos los landmarks de la mano, no solo la punta del dedo índice.

1.  **Crear un Historial Completo**:
    En `app.py`, localiza la sección de inicialización de `point_history` y añade una nueva cola (`deque`) para almacenar el historial de todos los landmarks.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py

    # ...
    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # NUEVO: Historial para todos los landmarks de la mano
    landmark_history = deque(maxlen=history_length) 
    # ...
    ```

2.  **Poblar el Nuevo Historial**:
    Dentro del bucle principal (`while True`), modifica la lógica para que siempre se guarde el estado completo de la mano (los 21 landmarks pre-procesados) en cada fotograma, independientemente de la pose detectada.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py

    # ... dentro de: if results.multi_hand_landmarks is not None:
    pre_processed_landmark_list = pre_process_landmark(landmark_list)
    
    # NUEVO: Guardar siempre el estado completo de la mano en el historial
    landmark_history.append(pre_processed_landmark_list)

    # El bloque que poblaba point_history condicionalmente ya no es necesario
    # y puede ser eliminado o comentado.
    # if hand_sign_id == 2: # Point gesture ...
    ```
    Asegúrate de manejar el caso en que no se detecta ninguna mano, añadiendo una lista de ceros para mantener la consistencia en la longitud de la secuencia.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py

    # ...
    else: # Si no se detecta mano
        point_history.append([0, 0])
        # NUEVO: Añadir ceros al historial completo (21 landmarks * 2 coordenadas)
        landmark_history.append([0] * (21 * 2))
    # ...
    ```

3.  **Adaptar el Guardado en CSV**:
    Modifica la función `logging_csv` para que guarde los datos del nuevo historial en un archivo CSV separado. Es **crucial** usar un nuevo archivo para no mezclar los datos, ya que el formato de entrada es diferente.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py

    def logging_csv(number, mode, landmark_list, landmark_history_list):
        # ... (el código para mode 1 no cambia) ...
        if mode == 2 and (0 <= number <= 9):
            # ¡IMPORTANTE!: Usa un nuevo archivo CSV
            csv_path = 'data/dynamic_gestures/dynamic_gestures_full.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                # Aplanamos la lista de listas (historial) en una sola fila para el CSV
                flat_landmark_history = list(itertools.chain.from_iterable(landmark_history_list))
                writer.writerow([number, *flat_landmark_history])
    return
    ```
    Finalmente, actualiza la llamada a `logging_csv` dentro del bucle principal para pasarle el nuevo historial.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py
    
    # ...
    logging_csv(number, mode, pre_processed_landmark_list, landmark_history)
    # ...
    ```
    **Acción Requerida**: Crea manualmente los nuevos archivos `data/dynamic_gestures/dynamic_gestures_full.csv` y `data/dynamic_gestures/dynamic_gestures_full_label.csv` antes de recolectar datos.

#### Paso 2: Adaptar y Re-entrenar el Modelo (`.ipynb`)

El modelo de Deep Learning debe ser ajustado para manejar la nueva estructura de datos de entrada, que ahora es significativamente más grande.

1.  **Abrir el Notebook**:
    Abre `dynamic_gestures_classification.ipynb`.

2.  **Actualizar Variables de Configuración**:
    *   Modifica la variable `dataset` para que apunte al nuevo archivo CSV.
    *   Define las nuevas dimensiones de la entrada.

    ```python
    # dynamic_gestures_classification.ipynb

    # Apuntar al nuevo dataset
    dataset = 'data/dynamic_gestures/dynamic_gestures_full.csv'
    
    # Actualizar las dimensiones de entrada
    TIME_STEPS = 16
    NUM_LANDMARKS = 21 * 2 # 42 valores por fotograma (21 puntos x, y)
    ```

3.  **Modificar la Carga de Datos**:
    Ajusta la función `np.loadtxt` para leer el número correcto de columnas (`TIME_STEPS * NUM_LANDMARKS`).

    ```python
    # dynamic_gestures_classification.ipynb

    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (TIME_STEPS * NUM_LANDMARKS) + 1)))
    ```

4.  **Ajustar la Arquitectura del Modelo**:
    La capa de entrada de la red neuronal debe coincidir con la nueva forma de los datos. Se recomienda **utilizar el modelo LSTM** (`use_lstm = True`), ya que es más adecuado para capturar dependencias temporales en secuencias largas.

    ```python
    # dynamic_gestures_classification.ipynb
    
    use_lstm = True
    model = None

    if use_lstm:
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * NUM_LANDMARKS, )),
            # La capa Reshape también debe ser actualizada
            tf.keras.layers.Reshape((TIME_STEPS, NUM_LANDMARKS), input_shape=(TIME_STEPS * NUM_LANDMARKS, )), 
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16, input_shape=[TIME_STEPS, NUM_LANDMARKS]),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])
    # ...
    ```

5.  **Re-entrenar el Modelo**:
    Ejecuta todas las celdas del notebook. Esto entrenará un nuevo modelo con los datos recolectados y lo guardará en los formatos `.h5` y `.tflite`. Asegúrate de actualizar las rutas de guardado (`model_save_path`, `tflite_save_path`) para no sobrescribir los modelos antiguos.

#### Paso 3: Integrar el Nuevo Clasificador en la Aplicación

Finalmente, utiliza el nuevo modelo entrenado para la clasificación en tiempo real.

1.  **Adaptar el Clasificador**:
    Aunque la lógica interna de `utils/dynamic_gestures_classifier.py` es genérica, es una buena práctica crear una copia, por ejemplo, `utils/full_dynamic_gestures_classifier.py`, para mantener el código organizado.

2.  **Integrar en `app.py`**:
    *   Importa y crea una instancia del nuevo clasificador, asegurándote de que cargue el modelo `.tflite` recién entrenado.
    *   En el bucle `while True`, después de poblar el `landmark_history`, aplana la estructura de datos y pásala al nuevo clasificador.

    ```python
    # C:\Users\Yefri Estiven Vera\Documents\GitHub\TESIS_GR\app.py

    # ...
    # Aplanar el historial para la entrada del modelo
    if len(landmark_history) == history_length:
        flat_landmark_history = list(itertools.chain.from_iterable(landmark_history))
        
        # Llamar al nuevo clasificador
        dynamic_gesture_id = full_point_history_classifier(flat_landmark_history)
        
        # ... (lógica para mostrar el resultado)
    # ...
    ```
    Ahora podrás mostrar simultáneamente la predicción del gesto estático (pose) y la del nuevo gesto dinámico (movimiento).

---

Con esta implementación, el sistema será capaz de reconocer una gama mucho más amplia y expresiva de gestos dinámicos, independientemente de la pose estática de la mano.