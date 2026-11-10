"""Genera señales sintéticas SOLO para verificar el pipeline (no son datos reales)."""
from pathlib import Path
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
t = np.arange(3 * SR) / SR
rng = np.random.default_rng(42)
signals = {
    "machinery": 0.23 * np.sin(2 * np.pi * 90 * t) + 0.09 * np.sin(2 * np.pi * 180 * t) + 0.03 * rng.normal(size=t.size),
    "speech": 0.22 * (1 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * (180 + 35 * np.sin(2 * np.pi * 2 * t)) * t),
    "alarms": 0.25 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0),
}
for category, x in signals.items():
    path = ROOT / "data" / "raw" / category
    path.mkdir(parents=True, exist_ok=True)
    filename = path / f"SYNTHETIC_DEMO_{category}.wav"
    wavfile.write(filename, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print(f"Creado (SINTÉTICO): {filename}")
