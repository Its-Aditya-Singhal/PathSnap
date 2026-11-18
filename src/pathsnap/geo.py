"""Geographic helpers: projection to metres, distances, bearings.

Trips are spread across the world (25+ UTM zones, both hemispheres), so
a single CRS is wrong. Each trip gets its own azimuthal-equidistant
projection centred on the trip; distortion is negligible at trip scale
(< ~75 km across).
"""
import numpy as np
from pyproj import CRS, Transformer


class LocalProjector:
    """Per-trip lon/lat (deg) <-> local metres (x east, y north)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs"
        )
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, lon, lat):
        return cls(np.mean(lon), np.mean(lat))

    def project(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def unproject(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371008.8
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all projected, broadcastable).

    Returns (distance_m, t) where t in [0,1] is the position of the
    closest point along AB.
    """
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_xy(dx, dy):
    """Bearing in degrees [0, 360) from a projected displacement (0 = north)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
