"""Caracterización reproducible de archivos WAV, sin estimar SPL físico."""
from pathlib import Path
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
CATEGORIES = ("machinery", "speech", "alarms")
BANDS = ((0, 300), (300, 1000), (1000, 3000), (3000, 8000))


def read_wav(path):
    sr, y = wavfile.read(path)
    if sr <= 0:
        raise ValueError("Frecuencia de muestreo inválida")
    if np.issubdtype(y.dtype, np.integer):
        if y.dtype == np.uint8:
            y = (y.astype(np.float64) - 128) / 128
        else:
            info = np.iinfo(y.dtype)
            y = y.astype(np.float64) / max(abs(info.min), abs(info.max))
    else:
        y = y.astype(np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or y.size < 32:
        raise ValueError("Audio vacío o demasiado corto")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio contiene NaN/Inf")
    return sr, y


def analyze(sr, y):
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    # Welch PSD: potencia por Hz; válido para ventanas de diferente duración.
    freq, psd = signal.welch(y, fs=sr, nperseg=min(2048, len(y)))
    valid = freq > 0
    dominant = float(freq[valid][np.argmax(psd[valid])]) if np.any(valid) else 0.0
    total = np.trapezoid(psd, freq)
    centroid = float(np.trapezoid(freq * psd, freq) / total) if total > 0 else 0.0
    result = {
        "duration_s": round(len(y) / sr, 4),
        "sample_rate_hz": sr,
        "rms": round(rms, 7),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "peak": round(peak, 7),
        "dominant_hz": round(dominant, 2),
        "spectral_centroid_hz": round(centroid, 2),
    }
    for low, high in BANDS:
        effective_high = min(high, sr / 2)
        mask = (freq >= low) & (freq <= effective_high)
        energy = np.trapezoid(psd[mask], freq[mask]) if np.count_nonzero(mask) >= 2 else 0
        result[f"band_{low}_{high}_pct"] = round(100 * energy / total, 2) if total > 0 else 0.0
    return result, freq, psd


def plot_audio(sr, y, freq, psd, title, dest):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(np.arange(len(y)) / sr, y, lw=0.55)
    axes[0].set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Forma de onda")
    axes[0].grid(alpha=.25)
    axes[1].plot(freq, 10 * np.log10(np.maximum(psd, 1e-16)))
    axes[1].set(xlabel="Frecuencia (Hz)", ylabel="PSD (dB/Hz, ref. digital)", title="Espectro de potencia (Welch)")
    axes[1].grid(alpha=.25)
    nperseg = min(512, len(y))
    f, t, Z = signal.stft(y, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    mesh = axes[2].pcolormesh(t, f, 20 * np.log10(np.maximum(abs(Z), 1e-8)), shading="auto", cmap="magma", vmin=-100, vmax=0)
    fig.colorbar(mesh, ax=axes[2], label="Magnitud (dB, ref. digital)")
    axes[2].set(xlabel="Tiempo (s)", ylabel="Frecuencia (Hz)", title="Espectrograma STFT", ylim=(0, min(4000, sr/2)))
    fig.suptitle(title)
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    rows = []
    for category in CATEGORIES:
        folder = RAW / category
        for path in sorted(folder.rglob("*.wav")) if folder.exists() else []:
            try:
                sr, y = read_wav(path)
                feats, freq, psd = analyze(sr, y)
                synthetic = "SYNTHETIC_DEMO" in path.name
                row = {"category": category, "filename": str(path.relative_to(RAW)), "synthetic_demo": synthetic, **feats}
                rows.append(row)
                safe = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)
                plot_audio(sr, y, freq, psd, f"{category} / {path.name}" , OUT / "plots" / f"{category}_{safe}.png")
                print(f"OK {path.name}: RMS {row['rms_dbfs']} dBFS, dominante {row['dominant_hz']} Hz")
            except (ValueError, OSError) as ex:
                print(f"ADVERTENCIA {path}: {ex}")
    if not rows:
        print("Sin archivos WAV. Agrega muestras a data/raw/{machinery,speech,alarms}/")
        return
    write_csv(OUT / "features.csv", rows)
    summary = []
    for category in CATEGORIES:
        clean = [r for r in rows if r["category"] == category and not r["synthetic_demo"]]
        if not clean:
            continue
        summary.append({"category": category, "n_real_files": len(clean), "mean_rms_dbfs": round(float(np.mean([r["rms_dbfs"] for r in clean])), 3), "mean_centroid_hz": round(float(np.mean([r["spectral_centroid_hz"] for r in clean])), 2)})
    write_csv(OUT / "category_summary.csv", summary)
    print(f"Procesados: {len(rows)} archivos. Resultados en: {OUT}")
    if not summary:
        print("Solo hay datos sintéticos: NO se generó resumen comparativo de clases reales.")


if __name__ == "__main__":
    main()
