
from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split
)
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "classification"

OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["machinery", "speech", "alarms"]

SR = 16000


def extract_features(path):

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    if len(y) < 512:
        raise ValueError("Audio demasiado corto")

    # Coeficientes MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # Centroide espectral
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    # Cruces por cero
    zcr = librosa.feature.zero_crossing_rate(y)

    # Energía RMS
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [centroid.mean()],
        [centroid.std()],
        [zcr.mean()],
        [rms.mean()]
    ])

    return features


X = []
y = []

for category in CATEGORIES:

    folder = DATA / category

    for path in sorted(folder.glob("*.wav")):

        try:
            features = extract_features(path)

            X.append(features)
            y.append(category)

        except Exception as error:
            print(f"Error en {path.name}: {error}")

X = np.asarray(X)
y = np.asarray(y)

if len(np.unique(y)) != 3:
    raise RuntimeError(
        "Se necesitan muestras de las 3 categorías"
    )

counts = [
    np.sum(y == category)
    for category in CATEGORIES
]

if min(counts) < 5:
    raise RuntimeError(
        "Se necesitan al menos 5 audios por categoría"
    )

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Modelo de clasificación
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Resultados
print("\nREPORTE DE CLASIFICACIÓN")

print(
    classification_report(
        y_test,
        predictions,
        labels=CATEGORIES,
        zero_division=0
    )
)

# Matriz de confusión
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=CATEGORIES,
    cmap="Blues"
)

plt.title("Matriz de confusión")

plt.tight_layout()

plt.savefig(
    OUT / "confusion_matrix.png",
    dpi=150
)

plt.show()
