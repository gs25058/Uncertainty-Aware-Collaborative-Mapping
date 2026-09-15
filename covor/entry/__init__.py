"""Phase 4: 3D occupancy -> a 2D entry map for a rescuer (DESIGN_entry_map.md).

Judgement happens on the grid; the vector render layer (render.py) only draws
what the grid already decided. Nothing in this package writes to covor/fusion.py,
covor/occupancy.py or covor/synth/ -- it consumes their output.
"""
from .config import EntryCfg   # noqa: F401
