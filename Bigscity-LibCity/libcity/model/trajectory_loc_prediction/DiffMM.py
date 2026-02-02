"""
NOTICE: DiffMM has been moved to the map_matching task.

DiffMM is a MAP MATCHING model (GPS-to-road segment matching), not a trajectory
location prediction model (next check-in prediction).

Please import DiffMM from:
    from libcity.model.map_matching.DiffMM import DiffMM

The model file is now located at:
    Bigscity-LibCity/libcity/model/map_matching/DiffMM.py

The config file is now located at:
    Bigscity-LibCity/libcity/config/model/map_matching/DiffMM.json
"""

# This file is deprecated - DiffMM moved to map_matching task
raise ImportError(
    "DiffMM has been moved to libcity.model.map_matching.DiffMM. "
    "DiffMM is a map matching model, not a trajectory location prediction model."
)
