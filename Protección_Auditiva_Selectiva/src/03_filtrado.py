
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "mixtures"
OUT = ROOT / "results" / "filtering"

OUT.mkdir(parents=True, exist_ok=True)

SR = 16000
N_FFT = 1024
HOP = 256

BETA = 1.0
MIN_GAIN = 0.10

mixtures = sorted(DATA.glob("*_mixture.wav"))

if not mixtures:
    raise RuntimeError(
        "Primero ejecuta 02_generar_mezclas.py"
    )


def power(x):
    return np.mean(np.asarray(x)**2)


for mix_path in mixtures:

    name = mix_path.name.replace(
        "_mixture.wav", ""
    )

    target_path = DATA / f"{name}_target.wav"
    noise_path = DATA / f"{name}_noise.wav"

    if not target_path.exists() or not noise_path.exists():
        print(f"Referencias faltantes: {name}")
        continue

    mixture, _ = librosa.load(
        mix_path, sr=SR, mono=True
    )

    target, _ = librosa.load(
        target_path, sr=SR, mono=True
    )

    noise, _ = librosa.load(
        noise_path, sr=SR, mono=True
    )

    # STFT de mezcla y ruido conocido
    X = librosa.stft(
        mixture,
        n_fft=N_FFT,
        hop_length=HOP
    )

    N = librosa.stft(
        noise,
        n_fft=N_FFT,
        hop_length=HOP
    )

    magnitude = np.abs(X)
    noise_magnitude = np.abs(N)

    # Máscara espectral
    gain = 1 - (
        BETA * noise_magnitude /
        (magnitude + 1e-10)
    )

    gain = np.clip(
        gain,
        MIN_GAIN,
        1.0
    )

    Y = gain * X

    # Reconstrucción
    filtered = librosa.istft(
        Y,
        hop_length=HOP,
        length=len(mixture)
    )

    # Métricas: error respecto de la señal objetivo
    input_error = mixture - target
    output_error = filtered - target

    snr_in = 10 * np.log10(
        power(target) /
        max(power(input_error), 1e-12)
    )

    snr_out = 10 * np.log10(
        power(target) /
        max(power(output_error), 1e-12)
    )

    improvement = snr_out - snr_in

    print(f"\n{name}")
    print(f"SNR entrada: {snr_in:.2f} dB")
    print(f"SNR salida: {snr_out:.2f} dB")
    print(f"Mejora: {improvement:.2f} dB")

    # Archivo procesado
    sf.write(
        OUT / f"{name}_filtered.wav",
        filtered,
        SR
    )

    # Comparación visual
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7)
    )

    original_db = librosa.amplitude_to_db(
        np.abs(X),
        ref=1.0
    )

    filtered_db = librosa.amplitude_to_db(
        np.abs(Y),
        ref=1.0
    )

    vmin = -100
    vmax = 0

    librosa.display.specshow(
        original_db,
        sr=SR,
        hop_length=HOP,
        x_axis="time",
        y_axis="linear",
        vmin=vmin,
        vmax=vmax,
        ax=axes[0]
    )

    librosa.display.specshow(
        filtered_db,
        sr=SR,
        hop_length=HOP,
        x_axis="time",
        y_axis="linear",
        vmin=vmin,
        vmax=vmax,
        ax=axes[1]
    )

    axes[0].set_title("Audio mezclado")
    axes[1].set_title("Audio filtrado")

    axes[0].set_ylim(0, 4000)
    axes[1].set_ylim(0, 4000)

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_comparison.png",
        dpi=150
    )

    plt.close()

print("\nFiltrado completado")
