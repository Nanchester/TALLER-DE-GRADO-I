
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "characterization"

OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["machinery", "speech", "alarms"]
SR = 16000

results = []

for category in CATEGORIES:

    folder = DATA / category

    if not folder.exists():
        print(f"No existe: {folder}")
        continue

    for audio_path in sorted(folder.glob("*.wav")):

        print(f"Analizando: {audio_path.name}")

        # Cargar audio
        y, sr = librosa.load(
            audio_path,
            sr=SR,
            mono=True
        )

        if len(y) == 0:
            continue

        # Duración
        duration = len(y) / sr

        # Energía RMS
        rms = np.sqrt(np.mean(y**2))

        # Nivel digital (no SPL)
        rms_dbfs = 20 * np.log10(
            max(rms, 1e-12)
        )

        # Transformada rápida de Fourier
        spectrum = np.abs(np.fft.rfft(y))
        frequencies = np.fft.rfftfreq(
            len(y), d=1 / sr
        )

        dominant_frequency = frequencies[
            np.argmax(spectrum)
        ]

        # Transformada de Fourier de tiempo corto
        stft = librosa.stft(
            y,
            n_fft=1024,
            hop_length=256
        )

        stft_db = librosa.amplitude_to_db(
            np.abs(stft),
            ref=1.0
        )

        # Centroide espectral
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        ).mean()

        # Guardar características
        results.append({
            "file": audio_path.name,
            "category": category,
            "duration_s": duration,
            "rms": rms,
            "rms_dbfs": rms_dbfs,
            "dominant_frequency_hz": dominant_frequency,
            "spectral_centroid_hz": centroid
        })

        # Gráficas
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 9)
        )

        librosa.display.waveshow(
            y, sr=sr, ax=axes[0]
        )
        axes[0].set_title("Señal temporal")

        axes[1].plot(
            frequencies,
            spectrum
        )
        axes[1].set_title("Espectro FFT")
        axes[1].set_xlabel("Frecuencia (Hz)")

        img = librosa.display.specshow(
            stft_db,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="linear",
            ax=axes[2]
        )

        axes[2].set_title("Espectrograma")
        fig.colorbar(
            img, ax=axes[2],
            label="Magnitud (dB)"
        )

        plt.tight_layout()

        output = OUT / (
            f"{category}_{audio_path.stem}.png"
        )

        plt.savefig(output, dpi=150)
        plt.close()

# Exportar resultados
if results:
    df = pd.DataFrame(results)

    df.to_csv(
        OUT / "features.csv",
        index=False
    )

    print("\nCaracterización completada")
    print(df)
else:
    print("No se encontraron audios WAV")
