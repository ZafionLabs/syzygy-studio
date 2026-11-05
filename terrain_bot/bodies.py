"""Real-world elevation references used to calibrate grayscale heightmaps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    name: str
    lat: float
    lon: float
    elevation_m: float
    source: str


@dataclass(frozen=True)
class Body:
    name: str
    mean_radius_km: float
    datum: str
    highest: Landmark
    lowest: Landmark

    @property
    def max_elevation_m(self) -> float:
        return self.highest.elevation_m

    @property
    def min_elevation_m(self) -> float:
        return self.lowest.elevation_m

    @property
    def relief_m(self) -> float:
        return self.max_elevation_m - self.min_elevation_m


EARTH = Body(
    name="Earth",
    mean_radius_km=6371.0,
    datum="mean sea level",
    highest=Landmark(
        name="Mount Everest",
        lat=27.9881,
        lon=86.9250,
        elevation_m=8848.86,
        source="China/Nepal joint survey, 2020",
    ),
    lowest=Landmark(
        name="Challenger Deep (Mariana Trench)",
        lat=11.3733,
        lon=142.5917,
        elevation_m=-10935.0,
        source="Greenaway et al. 2021, Deep-Sea Research I (10,935 +/- 6 m)",
    ),
)

MOON = Body(
    name="Moon",
    mean_radius_km=1737.4,
    datum="LOLA mean lunar radius (1737.4 km)",
    highest=Landmark(
        name="Selenean summit (Engel'gardt rim)",
        lat=5.4125,
        lon=-158.6335,
        elevation_m=10786.0,
        source="LRO LOLA / LROC NAC stereo, 2010",
    ),
    lowest=Landmark(
        name="Point Trieste (Antoniadi crater)",
        lat=-70.36,
        lon=-172.492,
        elevation_m=-9129.0,
        source="LRO LOLA / LROC QuickMap",
    ),
)

MARS = Body(
    name="Mars",
    mean_radius_km=3389.5,
    datum="MOLA areoid (Mars has no sea level)",
    highest=Landmark(
        name="Olympus Mons summit",
        lat=18.65,
        lon=-133.8,
        elevation_m=21229.0,
        source="MGS MOLA (Smith et al. 2001)",
    ),
    lowest=Landmark(
        name="Badwater crater (Hellas Planitia)",
        lat=-32.85,
        lon=62.21,
        elevation_m=-8201.0,
        source="MGS MOLA 463 m DEM",
    ),
)

BODIES = {"earth": EARTH, "moon": MOON, "mars": MARS}


def get_body(name: str) -> Body:
    try:
        return BODIES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown body {name!r}; choose from {sorted(BODIES)}") from exc
