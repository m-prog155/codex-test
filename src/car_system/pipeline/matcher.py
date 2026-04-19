from math import hypot

from car_system.types import Detection, PlateMatch


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _contains(outer: tuple[float, float, float, float], inner_center: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = outer
    cx, cy = inner_center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def match_plates_to_vehicles(vehicles: list[Detection], plates: list[Detection]) -> list[PlateMatch]:
    matches: list[PlateMatch] = []
    for plate in plates:
        plate_center = _center(plate.bbox)
        chosen: Detection | None = None

        containing = [vehicle for vehicle in vehicles if _contains(vehicle.bbox, plate_center)]
        if containing:
            chosen = min(containing, key=lambda vehicle: hypot(*(_center(vehicle.bbox)[0] - plate_center[0], _center(vehicle.bbox)[1] - plate_center[1])))
        elif vehicles:
            chosen = min(
                vehicles,
                key=lambda vehicle: hypot(
                    _center(vehicle.bbox)[0] - plate_center[0],
                    _center(vehicle.bbox)[1] - plate_center[1],
                ),
            )

        matches.append(PlateMatch(plate=plate, vehicle=chosen))
    return matches
