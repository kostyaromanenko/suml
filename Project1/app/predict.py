from pathlib import Path
import joblib
import numpy as np

SPECIES = ["setosa", "versicolor", "virginica"]

MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

def predict(features):

    features = np.array(features).reshape(1, -1)
    pred_idx = model.predict(features)[0]
    return SPECIES[pred_idx]
