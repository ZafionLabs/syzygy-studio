"""Find the most similar locations between two calibrated heightmaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .features import DEFAULT_SCALES_KM, FeatureSet, compute_features
from .heightmap import Heightmap, haversine_km

DEG_RADIUS = 180.0 / np.pi  # radius that makes haversine_km return degrees of arc

MODES = ("terrain", "elevation")


def default_weights(names: list[str]) -> dict[str, float]:
    weights = {}
    for n in names:
        if n == "elevation_m":
            weights[n] = 3.0
        elif n.startswith("slope_"):
            weights[n] = 0.5
        else:
            weights[n] = 1.0
    return weights


@dataclass
class MatchConfig:
    mode: str = "terrain"
    scales_km: tuple[float, ...] = DEFAULT_SCALES_KM
    weights: dict[str, float] | None = None
    min_separation_deg: float = 5.0
    candidates_per_point: int = 8
    max_samples: int = 250_000
    seed: int = 0


@dataclass
class SitePoint:
    body: str
    lat: float
    lon: float
    row: int
    col: int
    gray: float
    features: dict[str, float] = field(default_factory=dict)

    @property
    def elevation_m(self) -> float:
        return self.features["elevation_m"]


@dataclass
class Match:
    rank: int
    similarity: float
    distance: float
    a: SitePoint
    b: SitePoint

    @property
    def elevation_difference_m(self) -> float:
        return self.b.elevation_m - self.a.elevation_m

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elevation_difference_m"] = self.elevation_difference_m
        return d


class _Side:
    def __init__(self, hm: Heightmap, fs: FeatureSet, rows: np.ndarray, cols: np.ndarray):
        self.hm = hm
        self.fs = fs
        self.rows = rows
        self.cols = cols
        self.lats = hm.lats[rows]
        self.lons = hm.lons[cols]
        self.raw = fs.values[rows, cols]
        self.vectors: np.ndarray | None = None
        self.tree: cKDTree | None = None

    def point(self, i: int) -> SitePoint:
        r, c = int(self.rows[i]), int(self.cols[i])
        return self.site(r, c)

    def site(self, r: int, c: int) -> SitePoint:
        lat, lon = self.hm.latlon_of(r, c)
        return SitePoint(
            body=self.hm.body.name,
            lat=lat,
            lon=lon,
            row=r,
            col=c,
            gray=float(self.hm.gray[r, c]),
            features=self.fs.at(r, c),
        )


def _area_balanced_samples(fs: FeatureSet, hm: Heightmap, max_samples: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Pick pixels so each sample covers roughly equal ground area.

    Equirectangular rows near the poles are heavily oversampled, so columns are
    thinned by 1/cos(latitude).
    """
    rows, cols = [], []
    cos_lat = np.clip(np.cos(np.radians(hm.lats)), 1e-3, 1.0)
    for r in range(hm.shape[0]):
        step = max(1, int(round(1.0 / cos_lat[r])))
        offset = r % step
        c = np.arange(offset, hm.shape[1], step)
        c = c[fs.valid[r, c]]
        rows.append(np.full(c.size, r))
        cols.append(c)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    if rows.size > max_samples:
        keep = rng.choice(rows.size, size=max_samples, replace=False)
        keep.sort()
        rows, cols = rows[keep], cols[keep]
    return rows, cols


def _cell_ids(lats: np.ndarray, lons: np.ndarray, cell_deg: float) -> np.ndarray:
    band = np.floor((lats + 90.0) / cell_deg).astype(np.int64)
    cos_lat = np.clip(np.cos(np.radians(lats)), 1e-3, 1.0)
    col = np.floor((lons + 180.0) * cos_lat / cell_deg).astype(np.int64)
    return band * 100_000 + col


def _keep_top_per_group(groups: np.ndarray, per_group: int) -> np.ndarray:
    """``groups`` is already sorted by score; return a mask of the best ``per_group`` per id."""
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_inv)) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, sorted_inv.size]))
    rank_in_group = np.empty_like(order)
    rank_in_group[order] = np.arange(order.size) - group_start
    return rank_in_group < per_group


class TerrainMatcher:
    """Compare two heightmaps (e.g. Earth vs Moon) and report look-alike locations."""

    def __init__(self, hm_a: Heightmap, hm_b: Heightmap, config: MatchConfig | None = None):
        self.config = config or MatchConfig()
        if self.config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        rng = np.random.default_rng(self.config.seed)

        fs_a = compute_features(hm_a, self.config.scales_km)
        fs_b = compute_features(hm_b, self.config.scales_km)
        self.feature_names = fs_a.names

        self.a = _Side(hm_a, fs_a, *_area_balanced_samples(fs_a, hm_a, self.config.max_samples, rng))
        self.b = _Side(hm_b, fs_b, *_area_balanced_samples(fs_b, hm_b, self.config.max_samples, rng))
        if self.a.rows.size == 0 or self.b.rows.size == 0:
            raise ValueError("No valid terrain samples on one of the maps")

        if self.config.mode == "elevation":
            self.active = ["elevation_m"]
        else:
            self.active = list(self.feature_names)
        weights = default_weights(self.feature_names)
        if self.config.weights:
            weights.update(self.config.weights)
        self.weights = {n: weights[n] for n in self.active}

        idx = [self.feature_names.index(n) for n in self.active]
        pooled = np.concatenate([self.a.raw[:, idx], self.b.raw[:, idx]])
        self.mean = pooled.mean(axis=0)
        self.std = pooled.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        w = np.array([self.weights[n] for n in self.active])
        self._scale = np.sqrt(w / w.sum()) / self.std
        self._idx = idx

        for side in (self.a, self.b):
            side.vectors = self._embed(side.raw)
            side.tree = cKDTree(side.vectors)

    def _embed(self, raw: np.ndarray) -> np.ndarray:
        return (raw[..., self._idx] - self.mean) * self._scale

    @staticmethod
    def similarity(distance: np.ndarray | float):
        """Map a weighted RMS z-score distance to a 0-100 similarity score."""
        return 100.0 * np.exp(-np.asarray(distance))

    def _side(self, body: str) -> tuple[_Side, _Side]:
        key = body.lower()
        if key == self.a.hm.body.name.lower():
            return self.a, self.b
        if key == self.b.hm.body.name.lower():
            return self.b, self.a
        raise ValueError(f"Body {body!r} is not part of this comparison")

    def top_matches(self, n: int = 20) -> list[Match]:
        """Globally most similar pairs, spread out so they are not all in one spot."""
        cfg = self.config
        k = min(cfg.candidates_per_point, self.b.rows.size)
        dist, ib = self.b.tree.query(self.a.vectors, k=k)
        dist = np.atleast_2d(dist.reshape(len(self.a.vectors), -1))
        ib = np.atleast_2d(ib.reshape(len(self.a.vectors), -1))
        ia = np.repeat(np.arange(len(self.a.vectors)), dist.shape[1])
        dist, ib = dist.ravel(), ib.ravel()

        order = np.argsort(dist, kind="stable")
        ia, ib, dist = ia[order], ib[order], dist[order]

        sep = cfg.min_separation_deg
        per_cell = max(4, n)
        keep = _keep_top_per_group(_cell_ids(self.a.lats[ia], self.a.lons[ia], sep), per_cell)
        ia, ib, dist = ia[keep], ib[keep], dist[keep]
        keep = _keep_top_per_group(_cell_ids(self.b.lats[ib], self.b.lons[ib], sep), per_cell)
        ia, ib, dist = ia[keep], ib[keep], dist[keep]

        return self._greedy(ia, ib, dist, n, sep, separate_a=True)

    def _greedy(self, ia, ib, dist, n, sep, separate_a: bool) -> list[Match]:
        acc_a_lat, acc_a_lon, acc_b_lat, acc_b_lon = [], [], [], []
        matches: list[Match] = []
        for i_a, i_b, d in zip(ia, ib, dist):
            lat_b, lon_b = self.b.lats[i_b], self.b.lons[i_b]
            if acc_b_lat and np.min(haversine_km(lat_b, lon_b, np.array(acc_b_lat), np.array(acc_b_lon), DEG_RADIUS)) < sep:
                continue
            lat_a, lon_a = self.a.lats[i_a], self.a.lons[i_a]
            if separate_a and acc_a_lat and np.min(
                haversine_km(lat_a, lon_a, np.array(acc_a_lat), np.array(acc_a_lon), DEG_RADIUS)
            ) < sep:
                continue
            acc_a_lat.append(lat_a)
            acc_a_lon.append(lon_a)
            acc_b_lat.append(lat_b)
            acc_b_lon.append(lon_b)
            matches.append(
                Match(
                    rank=len(matches) + 1,
                    similarity=float(self.similarity(d)),
                    distance=float(d),
                    a=self.a.point(int(i_a)),
                    b=self.b.point(int(i_b)),
                )
            )
            if len(matches) >= n:
                break
        return matches

    def query(self, body: str, lat: float, lon: float, k: int = 5) -> tuple[SitePoint, list[Match]]:
        """Most similar places on the *other* body to the given coordinate."""
        src, dst = self._side(body)
        r, c = src.hm.pixel_of(lat, lon)
        if not src.fs.valid[r, c]:
            raise ValueError(f"({lat}, {lon}) has no valid terrain data on {src.hm.body.name}")
        origin = src.site(r, c)
        vec = self._embed(src.fs.values[r, c][None, :])
        n_cand = min(max(200, 50 * k), dst.rows.size)
        dist, idx = dst.tree.query(vec, k=n_cand)
        dist, idx = np.atleast_1d(dist[0]), np.atleast_1d(idx[0])

        results: list[Match] = []
        acc_lat, acc_lon = [], []
        sep = self.config.min_separation_deg
        for d, j in zip(dist, idx):
            lat_j, lon_j = dst.lats[j], dst.lons[j]
            if acc_lat and np.min(haversine_km(lat_j, lon_j, np.array(acc_lat), np.array(acc_lon), DEG_RADIUS)) < sep:
                continue
            acc_lat.append(lat_j)
            acc_lon.append(lon_j)
            results.append(
                Match(
                    rank=len(results) + 1,
                    similarity=float(self.similarity(d)),
                    distance=float(d),
                    a=origin,
                    b=dst.point(int(j)),
                )
            )
            if len(results) >= k:
                break
        return origin, results
