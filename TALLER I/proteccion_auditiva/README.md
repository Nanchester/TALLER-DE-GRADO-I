# Caracterización acústica preliminar — Taller I

Analiza archivos WAV de tres categorías: `machinery`, `speech` y `alarms`.
Calcula RMS, nivel digital dBFS, pico, centroide espectral, frecuencia dominante y energía por bandas. Guarda forma de onda, espectro y espectrograma, además de tablas CSV.

## Instalación en Windows (PowerShell)

```powershell
cd proteccion_auditiva
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, puedes ejecutar `.venv\Scripts\python.exe` en lugar de `python`.

## Primera prueba sin audios descargados

```powershell
python src/generate_demo.py
python src/characterize.py
```

**ATENCIÓN:** los WAV de demostración son señales **sintéticas**, diseñadas solo para comprobar el funcionamiento del programa. **No** se deben presentar como muestras reales de maquinaria, voz ni alarmas, ni usar para extraer conclusiones sobre diferencias entre clases.

## Trabajo con audios reales

1. Descarga un subconjunto de sonidos con licencia compatible (MIMII para maquinaria; FSD50K para alarmas; grabaciones de habla con permiso o Speech Commands).
2. Coloca los `.wav` en `data/raw/machinery`, `data/raw/speech` o `data/raw/alarms`.
3. Registra procedencia y condiciones en `data/metadata_template.csv` (una fila por audio). Conserva originales.
4. Ejecuta `python src/characterize.py`.
5. Revisa `results/features.csv`, `results/category_summary.csv` y los gráficos en `results/plots/`.

Para limpiar el análisis de demo, mueve sus WAV sintéticos fuera de `data/raw` antes de procesar grabaciones reales. No confundas los resultados.

## Notas metodológicas

- Se analiza la amplitud en unidades digitales, **no** SPL físico. Sin calibración de micrófono/sonómetro, dBFS no equivale a dB SPL.
- Si una grabación tiene varios sonidos, no le asignes una etiqueta exclusiva sin comprobarlo; las mezclas requieren etiquetas múltiples y un protocolo diferente.
- Mantén separados archivos de entrenamiento y prueba por grabación original/fuente para evitar fuga de datos.
- Los resultados dependen del dispositivo, grabación, ganancia, normalización y fuente del audio.
- El proyecto es experimental; ningún procesamiento o prototipo sin certificación debe utilizarse como protección auditiva industrial.

## Fuentes

- MIMII: https://zenodo.org/records/3384388
- FSD50K: https://zenodo.org/records/4060432
- Speech Commands: https://www.tensorflow.org/datasets/catalog/speech_commands

## Evidencia mínima para la exposición

- Número de muestras reales por clase y criterio de selección.
- Una forma de onda, un espectrograma y una FFT por clase (de muestras reales).
- Tabla de características, con interpretación y límites de medición.
- Diagrama de bloques del futuro sistema y próximos experimentos con mezclas.
