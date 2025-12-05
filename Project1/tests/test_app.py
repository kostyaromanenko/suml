
from pathlib import Path
import sys

# Make sure project root is on sys.path (so "app" can be imported)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predict import predict, SPECIES


def test_predict_returns_valid_species():
    # Example Iris measurements (typical Iris setosa values)
    features = [5.1, 3.5, 1.4, 0.2]

    species = predict(features)

    # Function should return a valid class (species) name
    assert species in SPECIES
