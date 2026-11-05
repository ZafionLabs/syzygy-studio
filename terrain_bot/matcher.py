"""Find the most similar locations between two elevation grids.

Every valid pixel of both maps is indexed (no subsampling), and nearest
neighbours are found with an exact KD-tree search, so the reported best match
is the same one a brute-force comparison against every pixel would return.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .features import DEFAULT_SCALES_KM, FeatureSet, compute_features
from .heightmap import Heightmap, haversine_km

DEG_RADIUS = 180.0 / np.pi  # radius that makes haversine_km return degrees of arc

MODES = ("terrain", "elevation")
SURFACES = ("any", "land", "underwater")
TIE_BREAK_WEIGHT = 1e-4


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
    breakdown: list[dict] = field(default_factory=list)

    @property
    def elevation_difference_m(self) -> float:
        return self.b.elevation_m - self.a.elevation_m

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elevation_difference_m"] = self.elevation_difference_m
        return d


def surface_mask(body_name: str, elevation: np.ndarray, surface: str) -> np.ndarray:
    """Earth-only filter on sea level; other bodies have no ocean, so it is a no-op there."""
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}")
    if surface == "any" or body_name != "Earth":
        return np.ones(elevation.shape, dtype=bool)
    return elevation < 0 if surface == "underwater" else elevation >= 0


class _Side:
    def __init__(self, hm: Heightmap, fs: FeatureSet):
        self.hm = hm
        self.fs = fs
        self.rows, self.cols = np.nonzero(fs.valid)
        self.lats = hm.lats[self.rows]
        self.lons = hm.lons[self.cols]
        self.raw = fs.values[self.rows, self.cols]
        self.vectors: np.ndarray | None = None
        self._trees: dict[str, tuple[np.ndarray, cKDTree]] = {}

    @property
    def size(self) -> int:
        return self.rows.size

    def subset(self, surface: str) -> tuple[np.ndarray, cKDTree]:
        """``(indices, tree)`` over the pixels allowed by ``surface``."""
        if surface not in self._trees:
            idx = np.flatnonzero(surface_mask(self.hm.body.name, self.raw[:, 0], surface))
            if idx.size == 0:
                raise ValueError(f"No {surface} pixels on {self.hm.body.name}")
            self._trees[surface] = (idx, cKDTree(self.vectors[idx]))
        return self._trees[surface]

    def point(self, i: int) -> SitePoint:
        return self.site(int(self.rows[i]), int(self.cols[i]))

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
    """Index any number of elevation grids (e.g. Earth, Moon, Mars) and report look-alike locations.

    Features are normalised with statistics pooled over every body in the
    index, so a body's vectors and KD-trees are shared by all pairings.
    """

    def __init__(
        self,
        maps: list[Heightmap] | tuple[Heightmap, ...],
        config: MatchConfig | None = None,
        feature_sets: dict[str, FeatureSet] | None = None,
    ):
        self.config = config or MatchConfig()
        if self.config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if len(maps) < 2:
            raise ValueError("Need at least two maps to compare")

        feature_sets = feature_sets or {}
        self.sides: dict[str, _Side] = {}
        for hm in maps:
            key = hm.body.name.lower()
            fs = feature_sets.get(key) or compute_features(hm, self.config.scales_km)
            self.sides[key] = _Side(hm, fs)
            if self.sides[key].size == 0:
                raise ValueError(f"No valid terrain pixels on {hm.body.name}")
        self.feature_names = next(iter(self.sides.values())).fs.names

        weights = default_weights(self.feature_names)
        if self.config.weights:
            weights.update(self.config.weights)
        if self.config.mode == "elevation":
            # Height decides; the other features only break ties between equal heights.
            weights = {n: (1.0 if n == "elevation_m" else TIE_BREAK_WEIGHT * w) for n, w in weights.items()}
        self.weights = weights
        self.active = list(self.feature_names)

        # Area-weighted statistics so the over-sampled polar rows don't dominate.
        pooled = np.concatenate([side.raw for side in self.sides.values()])
        area = np.concatenate([np.cos(np.radians(side.lats)) for side in self.sides.values()])
        area = np.clip(area, 1e-3, None)
        self.mean = np.average(pooled, axis=0, weights=area)
        self.std = np.sqrt(np.average((pooled - self.mean) ** 2, axis=0, weights=area))
        self.std[self.std < 1e-9] = 1.0
        w = np.array([self.weights[n] for n in self.feature_names])
        self.weight_share = w / w.sum()
        self._scale = np.sqrt(self.weight_share) / self.std

        for side in self.sides.values():
            side.vectors = self._embed(side.raw)

    @property
    def bodies(self) -> list[str]:
        return list(self.sides)

    @property
    def a(self) -> _Side:
        return self.sides[self.bodies[0]]

    @property
    def b(self) -> _Side:
        return self.sides[self.bodies[1]]

    def side(self, body: str) -> _Side:
        try:
            return self.sides[body.lower()]
        except KeyError:
            raise ValueError(f"Body {body!r} is not part of this comparison; choose from {self.bodies}") from None

    def default_target(self, body: str) -> str:
        """Earth pairs with the Moon by default; everything else pairs with Earth (or the first other body)."""
        body = body.lower()
        preferred = "moon" if body == "earth" else "earth"
        if preferred in self.sides:
            return preferred
        return next(b for b in self.bodies if b != body)

    def _embed(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.mean) * self._scale

    @staticmethod
    def similarity(distance):
        """Map the weighted RMS z-score distance to a 0-100 similarity score."""
        return 100.0 * np.exp(-np.asarray(distance))

    def breakdown(self, p: SitePoint, q: SitePoint) -> list[dict]:
        """Per-feature contribution to the distance between two points (contributions sum to distance^2)."""
        rows = []
        for i, n in enumerate(self.feature_names):
            diff = q.features[n] - p.features[n]
            rows.append(
                {
                    "feature": n,
                    "a": p.features[n],
                    "b": q.features[n],
                    "difference": diff,
                    "std": float(self.std[i]),
                    "z_difference": diff / float(self.std[i]),
                    "weight_share": float(self.weight_share[i]),
                    "contribution": float((diff * self._scale[i]) ** 2),
                }
            )
        return rows

    def _pair(self, body: str, target: str | None) -> tuple[_Side, _Side]:
        target = target or self.default_target(body)
        if target.lower() == body.lower():
            raise ValueError("Pick two different bodies to compare")
        return self.side(body), self.side(target)

    def _make_match(self, rank: int, d: float, p: SitePoint, q: SitePoint) -> Match:
        return Match(rank=rank, similarity=float(self.similarity(d)), distance=float(d), a=p, b=q,
                     breakdown=self.breakdown(p, q))

    def top_matches(self, n: int = 20, surface: str = "any", a: str | None = None, b: str | None = None) -> list[Match]:
        """Globally most similar pairs between bodies ``a`` and ``b``, spread out so they are not all in one spot.

        ``surface`` restricts the Earth side to land or underwater pixels.
        """
        side_a, side_b = self._pair(a or self.bodies[0], b or (self.bodies[1] if a is None else None))
        a_idx, _ = side_a.subset(surface)
        b_idx, b_tree = side_b.subset(surface)

        # Only close pairs can make the list, so search within a distance bound
        # estimated from a sample and widen it until enough matches survive.
        # Pairs inside the bound are found exactly, and the greedy selection
        # walks them in the same order as an unbounded search would.
        rng = np.random.default_rng(0)
        sample = rng.choice(a_idx, size=min(5000, a_idx.size), replace=False)
        sample_d, _ = b_tree.query(side_a.vectors[sample], k=1, workers=-1)
        pct = 2.0
        while True:
            bound = np.percentile(sample_d, pct) if pct < 100 else np.inf
            matches = self._top_within(side_a, side_b, a_idx, b_idx, b_tree, n, bound)
            if len(matches) >= n or not np.isfinite(bound):
                return matches
            pct = min(pct * 3, 100.0)

    def _top_within(self, side_a, side_b, a_idx, b_idx, b_tree, n: int, bound: float) -> list[Match]:
        cfg = self.config
        k = min(cfg.candidates_per_point, b_idx.size)
        dist, jb = b_tree.query(side_a.vectors[a_idx], k=k, distance_upper_bound=bound, workers=-1)
        dist = dist.reshape(a_idx.size, -1)
        jb = jb.reshape(a_idx.size, -1)
        found = np.isfinite(dist)
        ia = np.repeat(a_idx, dist.shape[1]).reshape(dist.shape)[found]
        ib = b_idx[jb[found]]
        dist = dist[found]

        order = np.argsort(dist, kind="stable")
        ia, ib, dist = ia[order], ib[order], dist[order]

        sep = cfg.min_separation_deg
        per_cell = max(4, n)
        keep = _keep_top_per_group(_cell_ids(side_a.lats[ia], side_a.lons[ia], sep), per_cell)
        ia, ib, dist = ia[keep], ib[keep], dist[keep]
        keep = _keep_top_per_group(_cell_ids(side_b.lats[ib], side_b.lons[ib], sep), per_cell)
        ia, ib, dist = ia[keep], ib[keep], dist[keep]

        acc_a, acc_b = [], []
        matches: list[Match] = []
        for i_a, i_b, d in zip(ia, ib, dist):
            pa = (side_a.lats[i_a], side_a.lons[i_a])
            pb = (side_b.lats[i_b], side_b.lons[i_b])
            if _too_close(pb, acc_b, sep) or _too_close(pa, acc_a, sep):
                continue
            acc_a.append(pa)
            acc_b.append(pb)
            matches.append(self._make_match(len(matches) + 1, d, side_a.point(int(i_a)), side_b.point(int(i_b))))
            if len(matches) >= n:
                break
        return matches

    def query(
        self, body: str, lat: float, lon: float, k: int = 5, surface: str = "any", target: str | None = None
    ) -> tuple[SitePoint, list[Match]]:
        """Most similar places on ``target`` (default: :meth:`default_target`) to the given coordinate."""
        src, dst = self._pair(body, target)
        r, c = src.hm.pixel_of(lat, lon)
        if not src.fs.valid[r, c]:
            raise ValueError(f"({lat}, {lon}) has no valid terrain data on {src.hm.body.name}")
        origin = src.site(r, c)
        vec = self._embed(src.fs.values[r, c][None, :])
        idx, tree = dst.subset(surface)

        sep = self.config.min_separation_deg
        n_cand = min(max(200, 50 * k), idx.size)
        while True:
            dist, j = tree.query(vec, k=n_cand)
            dist, j = np.atleast_1d(dist[0]), idx[np.atleast_1d(j[0])]
            results, accepted = [], []
            for d, jj in zip(dist, j):
                p = (dst.lats[jj], dst.lons[jj])
                if _too_close(p, accepted, sep):
                    continue
                accepted.append(p)
                results.append(self._make_match(len(results) + 1, d, origin, dst.point(int(jj))))
                if len(results) >= k:
                    break
            if len(results) >= k or n_cand >= idx.size:
                return origin, results
            n_cand = min(n_cand * 4, idx.size)

    def brute_force_best(
        self, body: str, lat: float, lon: float, surface: str = "any", target: str | None = None
    ) -> tuple[int, float]:
        """Reference implementation: compare against every pixel explicitly (used to verify the KD-tree)."""
        src, dst = self._pair(body, target)
        r, c = src.hm.pixel_of(lat, lon)
        vec = self._embed(src.fs.values[r, c][None, :])[0]
        idx, _ = dst.subset(surface)
        d = np.sqrt(((dst.vectors[idx] - vec) ** 2).sum(axis=1))
        best = int(np.argmin(d))
        return int(idx[best]), float(d[best])

    def search_size(self, target: str, surface: str = "any") -> int:
        return int(self.side(target).subset(surface)[0].size)


def _too_close(p: tuple[float, float], accepted: list[tuple[float, float]], sep: float) -> bool:
    if not accepted:
        return False
    arr = np.asarray(accepted)
    return bool(np.min(haversine_km(p[0], p[1], arr[:, 0], arr[:, 1], DEG_RADIUS)) < sep)
