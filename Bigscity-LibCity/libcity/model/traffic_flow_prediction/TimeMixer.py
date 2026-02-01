# This file has been deprecated and removed.
# The correct TimeMixer implementation is located at:
# libcity/model/traffic_speed_prediction/TimeMixer.py
#
# This file was removed to fix duplicate registration issue.
# The traffic_flow_prediction version had a shape mismatch bug in calculate_loss.

raise ImportError(
    "TimeMixer has been moved to traffic_speed_prediction. "
    "This file should be deleted. Please use the version from "
    "libcity.model.traffic_speed_prediction.TimeMixer instead."
)
