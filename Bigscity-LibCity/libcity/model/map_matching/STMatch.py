"""
ST-Match Algorithm (FMM variant) for LibCity

This module implements the STMATCH algorithm from the FMM library.
STMATCH computes shortest paths on-the-fly instead of using precomputed UBODT,
making it more suitable for dynamic networks or when memory is constrained.

Key differences from FMM:
- No precomputation step (no UBODT)
- Uses bounded Dijkstra with delta = vmax * factor * delta_time
- Better for sparse trajectories with large time gaps

Reference:
    Can Yang and Gyozo Gidofalvi. "Fast map matching, an algorithm integrating
    hidden Markov model with precomputation." International Journal of Geographical
    Information Science 32.3 (2018): 547-570.

Original C++ implementation: https://github.com/cyang-kth/fmm

Adapted for LibCity framework.
"""

import math
import heapq
import networkx as nx
import numpy as np
from logging import getLogger
from typing import Dict, List, Tuple, Optional, Set

from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
from libcity.utils.GPS_utils import radian2angle, R_EARTH, angle2radian, dist


class Candidate:
    """
    Represents a candidate road segment for a GPS point.
    """

    def __init__(self, edge: Tuple, offset: float, dist: float,
                 point: Tuple[float, float], edge_length: float,
                 source_idx: int = -1):
        self.edge = edge
        self.offset = offset
        self.dist = dist
        self.point = point
        self.edge_length = edge_length
        # Index for dummy node in composite graph
        self.index = source_idx

    def __repr__(self):
        return f"Candidate(edge={self.edge}, dist={self.dist:.2f})"


class TGNode:
    """
    A node in the transition graph for HMM-based map matching.
    """

    def __init__(self, candidate: Candidate, ep: float):
        self.candidate = candidate
        self.prev: Optional[TGNode] = None
        self.ep = ep
        self.tp = 0.0
        self.cumu_prob = -float('inf')
        self.sp_dist = 0.0


class STMatch(AbstractTraditionModel):
    """
    STMATCH Algorithm (On-the-fly shortest path computation variant of FMM).

    This implementation computes shortest paths on-the-fly using bounded
    Dijkstra, making it more memory efficient than FMM with UBODT but
    potentially slower for dense trajectories.

    Key Parameters:
        k: Number of candidate edges per GPS point (default: 8)
        r: Search radius for candidate edges in meters (default: 300)
        gps_error: GPS measurement error standard deviation (default: 50)
        vmax: Maximum vehicle speed in m/s (default: 30)
        factor: Factor for search bound computation (default: 1.5)
        reverse_tolerance: Allowed proportion of reverse movement (default: 0.0)
    """

    def __init__(self, config, data_feature):
        """
        Initialize STMATCH model.

        Args:
            config: Configuration dictionary with model parameters
            data_feature: Data feature dictionary with dataset properties
        """
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # STMATCH algorithm parameters
        self.k = config.get('k', 8)
        self.r = config.get('r', 300)
        self.gps_error = config.get('gps_error', 50)
        self.vmax = config.get('vmax', 30)  # Maximum speed in m/s
        self.factor = config.get('factor', 1.5)  # Search bound factor
        self.reverse_tolerance = config.get('reverse_tolerance', 0.0)

        # Data parameters
        self.with_time = data_feature.get('with_time', True)

        # Data cache
        self.rd_nwk: Optional[nx.DiGraph] = None
        self.usr_id = None
        self.traj_id = None
        self.trajectory = None
        self.res_dct: Dict = {}

        # Spatial indexing parameters
        self.lon_r = None
        self.lat_r = None

        # Node index counter for composite graph
        self._node_counter = 0

    def run(self, data: Dict) -> Dict:
        """
        Run STMATCH map matching on trajectories.

        Args:
            data: Dictionary containing:
                - 'rd_nwk': NetworkX graph of road network
                - 'trajectory': Dict of user_id -> traj_id -> trajectory data

        Returns:
            Dictionary of matching results
        """
        self.rd_nwk = data['rd_nwk']
        trajectory = data['trajectory']

        # Set spatial search parameters
        first_node = list(self.rd_nwk.nodes)[0]
        self._set_lon_lat_radius(
            self.rd_nwk.nodes[first_node]['lon'],
            self.rd_nwk.nodes[first_node]['lat']
        )

        # Get base node count for composite graph indexing
        self._node_counter = len(self.rd_nwk.nodes)

        # Process each trajectory
        for usr_id, usr_value in trajectory.items():
            self.usr_id = usr_id
            for traj_id, value in usr_value.items():
                self._logger.info(f'STMATCH: begin map matching, usr_id:{usr_id} traj_id:{traj_id}')
                self.traj_id = traj_id
                self.trajectory = value
                self._run_one_trajectory()
                self._logger.info(f'STMATCH: finish map matching, usr_id:{usr_id} traj_id:{traj_id}')

        return self.res_dct

    def _set_lon_lat_radius(self, lon: float, lat: float):
        """Compute search radius in degrees."""
        self.lat_r = radian2angle(self.r / R_EARTH)
        r_prime = R_EARTH * math.cos(angle2radian(lat))
        self.lon_r = radian2angle(self.r / r_prime)

    def _run_one_trajectory(self):
        """Run STMATCH algorithm for a single trajectory."""
        # Step 1: Get candidates with composite graph indices
        candidates_per_point, composite_graph = self._get_candidates_with_graph()

        if not candidates_per_point or all(len(c) == 0 for c in candidates_per_point):
            self._logger.warning('No candidates found for trajectory')
            self._store_empty_result()
            return

        # Step 2: Build transition graph
        tg_layers = self._build_transition_graph(candidates_per_point)

        # Step 3: Update using on-the-fly shortest path computation
        self._update_transition_graph(tg_layers, composite_graph)

        # Step 4: Backtrack
        optimal_path = self._backtrack(tg_layers)

        # Step 5: Store results
        self._store_result(optimal_path)

    def _get_candidates_with_graph(self) -> Tuple[List[List[Candidate]], nx.DiGraph]:
        """
        Find candidates and build composite graph.

        The composite graph includes the original road network plus
        dummy nodes for each candidate to handle partial edge traversal.

        Returns:
            Tuple of (candidates_per_point, composite_graph)
        """
        traj_lon_lat = self.trajectory[:, 1:3]
        candidates_per_point = []

        # Create composite graph (copy of road network)
        composite_graph = self.rd_nwk.copy()

        current_idx = len(self.rd_nwk.nodes)

        for i in range(traj_lon_lat.shape[0]):
            lon, lat = traj_lon_lat[i, :]
            point_candidates = []

            for edge in self.rd_nwk.edges:
                source, target = edge[:2]

                src_lat = self.rd_nwk.nodes[source]['lat']
                src_lon = self.rd_nwk.nodes[source]['lon']
                tgt_lat = self.rd_nwk.nodes[target]['lat']
                tgt_lon = self.rd_nwk.nodes[target]['lon']

                if not self._edge_in_radius(lon, lat, src_lon, src_lat, tgt_lon, tgt_lat):
                    continue

                distance, offset, proj_point = self._point_to_edge_distance(
                    lon, lat, src_lon, src_lat, tgt_lon, tgt_lat
                )

                if distance <= self.r:
                    edge_data = self.rd_nwk.get_edge_data(source, target)
                    edge_length = edge_data.get('distance', edge_data.get('length', 0))

                    candidate = Candidate(
                        edge=(source, target),
                        offset=offset,
                        dist=distance,
                        point=proj_point,
                        edge_length=edge_length,
                        source_idx=current_idx
                    )
                    point_candidates.append(candidate)

                    # Add dummy node to composite graph
                    composite_graph.add_node(current_idx, lat=proj_point[0], lon=proj_point[1])

                    # Add edges from dummy node TO network nodes
                    # Edge to target with remaining distance
                    remaining = edge_length - offset if edge_length > 0 else 0
                    composite_graph.add_edge(current_idx, target, distance=remaining)

                    # Add edge FROM source node TO dummy node (for paths entering this edge)
                    # This allows paths: dummy_A -> target_A -> ... -> source_B -> dummy_B
                    composite_graph.add_edge(source, current_idx, distance=offset)

                    # Handle reverse tolerance
                    if self.reverse_tolerance > 0:
                        composite_graph.add_edge(current_idx, source,
                                                distance=offset * (1 - self.reverse_tolerance))

                    current_idx += 1

            point_candidates.sort(key=lambda c: c.dist)
            candidates_per_point.append(point_candidates[:self.k])

        return candidates_per_point, composite_graph

    def _edge_in_radius(self, lon: float, lat: float,
                        src_lon: float, src_lat: float,
                        tgt_lon: float, tgt_lat: float) -> bool:
        """Check if edge is within search radius."""
        if (lat - self.lat_r <= src_lat <= lat + self.lat_r and
            lon - self.lon_r <= src_lon <= lon + self.lon_r):
            return True
        if (lat - self.lat_r <= tgt_lat <= lat + self.lat_r and
            lon - self.lon_r <= tgt_lon <= lon + self.lon_r):
            return True

        mid_lat = (src_lat + tgt_lat) / 2
        mid_lon = (src_lon + tgt_lon) / 2
        if (lat - self.lat_r <= mid_lat <= lat + self.lat_r and
            lon - self.lon_r <= mid_lon <= lon + self.lon_r):
            return True

        return False

    def _point_to_edge_distance(self, lon: float, lat: float,
                                src_lon: float, src_lat: float,
                                tgt_lon: float, tgt_lat: float
                                ) -> Tuple[float, float, Tuple[float, float]]:
        """Compute perpendicular distance from point to edge."""
        lat_r = angle2radian(lat)
        lon_r = angle2radian(lon)
        src_lat_r = angle2radian(src_lat)
        src_lon_r = angle2radian(src_lon)
        tgt_lat_r = angle2radian(tgt_lat)
        tgt_lon_r = angle2radian(tgt_lon)

        a = dist(lat_r, lon_r, src_lat_r, src_lon_r)
        b = dist(lat_r, lon_r, tgt_lat_r, tgt_lon_r)
        c = dist(src_lat_r, src_lon_r, tgt_lat_r, tgt_lon_r)

        if c == 0:
            return a, 0, (src_lat, src_lon)

        if b * b > a * a + c * c:
            return a, 0, (src_lat, src_lon)

        if a * a > b * b + c * c:
            return b, c, (tgt_lat, tgt_lon)

        p = (a + b + c) / 2
        area_sq = p * abs(p - a) * abs(p - b) * abs(p - c)
        if area_sq < 0:
            area_sq = 0
        area = math.sqrt(area_sq)
        distance = 2 * area / c

        offset = math.sqrt(max(0, a * a - distance * distance))

        t = offset / c if c > 0 else 0
        proj_lat = src_lat + t * (tgt_lat - src_lat)
        proj_lon = src_lon + t * (tgt_lon - src_lon)

        return distance, offset, (proj_lat, proj_lon)

    def _build_transition_graph(self, candidates_per_point: List[List[Candidate]]) -> List[List[TGNode]]:
        """Build transition graph with emission probabilities."""
        layers = []

        for candidates in candidates_per_point:
            layer = []
            for candidate in candidates:
                ep = self._calc_emission_prob(candidate.dist)
                node = TGNode(candidate, ep)
                layer.append(node)
            layers.append(layer)

        if layers and layers[0]:
            for node in layers[0]:
                node.cumu_prob = math.log(node.ep) if node.ep > 0 else -float('inf')

        return layers

    def _calc_emission_prob(self, distance: float) -> float:
        """Calculate emission probability."""
        a = distance / self.gps_error
        return math.exp(-0.5 * a * a)

    def _calc_transition_prob(self, sp_dist: float, eu_dist: float) -> float:
        """Calculate transition probability."""
        if sp_dist == 0:
            return 1.0
        if sp_dist == float('inf'):
            return 0.0
        return min(1.0, eu_dist / sp_dist)

    def _update_transition_graph(self, layers: List[List[TGNode]], composite_graph: nx.DiGraph):
        """Update transition graph using bounded Dijkstra."""
        traj_data = self.trajectory

        for i in range(len(layers) - 1):
            if not layers[i] or not layers[i + 1]:
                continue

            # Compute search bound delta
            if self.with_time and traj_data.shape[1] > 3:
                time_i = traj_data[i, 3]
                time_j = traj_data[i + 1, 3]
                duration = abs(time_j - time_i)
                delta = self.factor * self.vmax * duration
            else:
                # Use Euclidean distance as fallback
                lon1, lat1 = traj_data[i, 1:3]
                lon2, lat2 = traj_data[i + 1, 1:3]
                eu_dist = dist(angle2radian(lat1), angle2radian(lon1),
                              angle2radian(lat2), angle2radian(lon2))
                delta = eu_dist * self.factor * 4

            # Compute Euclidean distance
            lon1, lat1 = traj_data[i, 1:3]
            lon2, lat2 = traj_data[i + 1, 1:3]
            eu_dist = dist(angle2radian(lat1), angle2radian(lon1),
                          angle2radian(lat2), angle2radian(lon2))

            self._update_layer(layers[i], layers[i + 1], composite_graph, eu_dist, delta)

    def _update_layer(self, layer_a: List[TGNode], layer_b: List[TGNode],
                      composite_graph: nx.DiGraph, eu_dist: float, delta: float):
        """Update transitions using bounded Dijkstra."""
        # Get target indices
        target_indices = [node.candidate.index for node in layer_b]

        for node_a in layer_a:
            source_idx = node_a.candidate.index

            # Run bounded Dijkstra
            distances = self._bounded_dijkstra(composite_graph, source_idx, target_indices, delta)

            for j, node_b in enumerate(layer_b):
                sp_dist = distances[j]
                tp = self._calc_transition_prob(sp_dist, eu_dist)

                if node_a.cumu_prob > -float('inf') and tp > 0 and node_b.ep > 0:
                    cumu_prob = node_a.cumu_prob + math.log(tp) + math.log(node_b.ep)
                else:
                    cumu_prob = -float('inf')

                if cumu_prob > node_b.cumu_prob:
                    node_b.cumu_prob = cumu_prob
                    node_b.prev = node_a
                    node_b.tp = tp
                    node_b.sp_dist = sp_dist

    def _bounded_dijkstra(self, graph: nx.DiGraph, source: int,
                          targets: List[int], delta: float) -> List[float]:
        """
        Run bounded Dijkstra to find distances to multiple targets.

        Args:
            graph: Composite graph
            source: Source node index
            targets: List of target node indices
            delta: Upper bound for search

        Returns:
            List of distances to each target
        """
        distances = {source: 0}
        heap = [(0, source)]
        visited = set()
        target_set = set(targets)
        found = set()

        while heap and len(found) < len(targets):
            d, u = heapq.heappop(heap)

            if u in visited:
                continue
            visited.add(u)

            if d > delta:
                break

            if u in target_set:
                found.add(u)

            for v in graph.neighbors(u):
                edge_data = graph.get_edge_data(u, v)
                weight = edge_data.get('distance', edge_data.get('weight', 1))
                new_dist = d + weight

                if new_dist <= delta and (v not in distances or new_dist < distances[v]):
                    distances[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))

        # Return distances for all targets
        result = []
        for t in targets:
            result.append(distances.get(t, float('inf')))

        return result

    def _backtrack(self, layers: List[List[TGNode]]) -> List[TGNode]:
        """Backtrack to find optimal path."""
        if not layers:
            return []

        best_node = None
        best_prob = -float('inf')

        for layer in reversed(layers):
            if not layer:
                continue
            for node in layer:
                if node.cumu_prob > best_prob:
                    best_prob = node.cumu_prob
                    best_node = node
            if best_node is not None:
                break

        if best_node is None or best_prob == -float('inf'):
            return []

        path = []
        current = best_node
        while current is not None:
            path.append(current)
            current = current.prev

        path.reverse()
        return path

    def _store_result(self, optimal_path: List[TGNode]):
        """Store matching result."""
        if not optimal_path:
            self._store_empty_result()
            return

        n_points = len(self.trajectory)
        path_idx = 0
        results = []

        for i in range(n_points):
            dyna_id = int(self.trajectory[i, 0])

            if path_idx < len(optimal_path):
                edge = optimal_path[path_idx].candidate.edge
                geo_id = self.rd_nwk.edges[edge].get('geo_id', edge)
                path_idx += 1
            else:
                geo_id = None

            if self.with_time and self.trajectory.shape[1] > 3:
                time_val = self.trajectory[i, 3]
                results.append([dyna_id, geo_id, time_val])
            else:
                results.append([dyna_id, geo_id])

        res_all = np.array(results, dtype=object)

        if self.usr_id in self.res_dct:
            self.res_dct[self.usr_id][self.traj_id] = res_all
        else:
            self.res_dct[self.usr_id] = {self.traj_id: res_all}

    def _store_empty_result(self):
        """Store empty result when matching fails."""
        n_points = len(self.trajectory)
        results = []

        for i in range(n_points):
            dyna_id = int(self.trajectory[i, 0])
            if self.with_time and self.trajectory.shape[1] > 3:
                time_val = self.trajectory[i, 3]
                results.append([dyna_id, None, time_val])
            else:
                results.append([dyna_id, None])

        res_all = np.array(results, dtype=object)

        if self.usr_id in self.res_dct:
            self.res_dct[self.usr_id][self.traj_id] = res_all
        else:
            self.res_dct[self.usr_id] = {self.traj_id: res_all}
