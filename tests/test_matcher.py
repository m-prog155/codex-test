from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.pipeline.matcher import match_plates_to_vehicles
from car_system.types import Detection


def test_match_plates_to_vehicles_prefers_containing_vehicle() -> None:
    vehicles = [
        Detection(label="car", confidence=0.9, bbox=(10, 10, 120, 120)),
        Detection(label="truck", confidence=0.8, bbox=(150, 10, 300, 180)),
    ]
    plates = [
        Detection(label="plate", confidence=0.95, bbox=(40, 80, 90, 105)),
        Detection(label="plate", confidence=0.92, bbox=(190, 120, 245, 150)),
    ]

    matches = match_plates_to_vehicles(vehicles, plates)

    assert len(matches) == 2
    assert matches[0].vehicle is vehicles[0]
    assert matches[0].plate is plates[0]
    assert matches[1].vehicle is vehicles[1]
    assert matches[1].plate is plates[1]
