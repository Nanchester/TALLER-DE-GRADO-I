
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "raw"
OUT = ROOT / "data" / "mixtures"

OUT.mkdir(parents=True, exist_ok=True)

SR = 16000
DURATION = 3
N = SR * DURATION

SNR_LEVELS = [-5, 0, 5]


def load_segment(path):
    y, _ = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    if len(y) < N:
        y = np.pad(
            y,
            (0, N - len(y))
        )

    return y[:N]


def rms(x):
    return np.sqrt(np.mean(x**2))


def create_mixture(signal, noise, snr_db):

    # Igualar el ruido a la SNR deseada
    signal_rms = rms(signal)
    noise_rms = rms(noise)

    if signal_rms < 1e-10 or noise_rms < 1e-10:
        raise ValueError("Señal o ruido sin energía")

    target_noise_rms = (
        signal_rms / (10 ** (snr_db / 20))
    )

    noise_scaled = noise * (
        target_noise_rms / noise_rms
    )

    mixture = signal + noise_scaled

    # Un único factor para evitar clipping.
    # También se aplica a las fuentes de referencia.
    peak = np.max(np.abs(mixture))
    scale = min(1.0, 0.95 / max(peak, 1e-12))

    return (
        mixture * scale,
        signal * scale,
        noise_scaled * scale
    )


machinery_files = sorted(
    (DATA / "machinery").glob("*.wav")
)

speech_files = sorted(
    (DATA / "speech").glob("*.wav")
)

alarm_files = sorted(
    (DATA / "alarms").glob("*.wav")
)

if not machinery_files:
    raise RuntimeError("Faltan audios de maquinaria")

noise = load_segment(machinery_files[0])

for category, files in [
    ("speech", speech_files),
    ("alarms", alarm_files)
]:

    if not files:
        print(f"Sin audios de {category}")
        continue

    source = load_segment(files[0])

    for snr in SNR_LEVELS:

        mix, clean, scaled_noise = create_mixture(
            source, noise, snr
        )

        name = f"{category}_snr_{snr:+d}dB"

        sf.write(
            OUT / f"{name}_mixture.wav",
            mix,
            SR
        )

        sf.write(
            OUT / f"{name}_target.wav",
            clean,
            SR
        )

        sf.write(
            OUT / f"{name}_noise.wav",
            scaled_noise,
            SR
        )

        print(f"Mezcla generada: {name}")

print("Experimento completado")
