"""Per-trip preparation shared by labelling, training, decoding and the API."""
from dataclasses import dataclass

import numpy as np

from .geo import LocalProjector
from .io import Trip
from .network import RoadNetwork


@dataclass
class TripContext:
    trip: Trip
    proj: LocalProjector
    net: RoadNetwork
    px: np.ndarray
    py: np.ndarray
    ts: np.ndarray


def prepare(trip: Trip) -> TripContext:
    tr = trip.track
    proj = LocalProjector.for_track(tr.lon.values, tr.lat.values)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.project(tr.lon.values, tr.lat.values)
    return TripContext(trip, proj, net, px, py, tr.t.to_numpy())
