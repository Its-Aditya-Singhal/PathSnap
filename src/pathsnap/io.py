"""Loading raw PathSnap trip files.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations)
  <id>.nodes  lon lat            (row index = node_id)
  <id>.arcs   from_node to_node  (row index = arc_id, directed)
  <id>.route  arc_id             (ordered reference route; optional at inference)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_ROOT, PARTITIONS


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame            # columns: lon, lat, t
    nodes: pd.DataFrame            # columns: lon, lat (index = node_id)
    arcs: pd.DataFrame             # columns: u, v (index = arc_id)
    route: Optional[np.ndarray]    # ordered arc_ids, or None if not supplied


def _read(path: Path, names, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(trip_dir, partition: str = "", with_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read(trip_dir / f"{tid}.track", ["lon", "lat", "t"], float)
    nodes = _read(trip_dir / f"{tid}.nodes", ["lon", "lat"], float)
    arcs = _read(trip_dir / f"{tid}.arcs", ["u", "v"], int)
    route = None
    route_path = trip_dir / f"{tid}.route"
    if with_route and route_path.exists():
        route = _read(route_path, ["arc"], int)["arc"].to_numpy()
    return Trip(tid, partition or trip_dir.parent.name, track, nodes, arcs, route)


def list_trips(partition: str, data_root: Path = DATA_ROOT) -> list[Path]:
    part_dir = Path(data_root) / partition
    return sorted(p for p in part_dir.iterdir() if p.is_dir())


def iter_trips(partitions=PARTITIONS, data_root: Path = DATA_ROOT, with_route=True):
    for part in partitions:
        for d in list_trips(part, data_root):
            yield load_trip(d, part, with_route=with_route)
