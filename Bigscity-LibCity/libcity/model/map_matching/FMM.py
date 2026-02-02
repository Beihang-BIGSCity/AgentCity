"""
Fast Map Matching (FMM) Algorithm for LibCity

This module implements the Fast Map Matching algorithm based on the FMM paper and C++ implementation.
FMM uses a precomputed Upper Bounded Origin Destination Table (UBODT) for efficient shortest path queries.

Reference:
    Can Yang and Gyozo Gidofalvi. "Fast map matching, an algorithm integrating
    hidden Markov model with precomputation." International Journal of Geographical
    Information Science 32.3 (2018): 547-570.

Original C++ implementation: https://github.com/cyang-kth/fmm

Adapted for LibCity framework by following the AbstractTraditionModel conventions.
"""

import math
import heapq
import networkx as nx
import numpy as np
from logging import getLogger
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
from libcity.utils.GPS_utils import radian2angle, R_EARTH, angle2radian, dist


class Candidate:
    """
    Represents a candidate road segment for a GPS point.

    Attributes:
        edge: Tuple of (source, target) node IDs defining the edge
        offset: Distance from edge start to the projected point on edge
        dist: Perpendicular distance from GPS point to edge (GPS error)
        point: Projected point coordinates on edge (lat, lon)
        edge_length: Total length of the edge
    """

    def __init__(self, edge: Tuple, offset: float, dist: float,
                 point: Tuple[float, float], edge_length: float):
        self.edge = edge
        self.offset = offset
        self.dist = dist
        self.point = point
        self.edge_length = edge_length

    def __repr__(self):
        return f"Candidate(edge={self.edge}, dist={self.dist:.2f}, offset={self.offset:.2f})"


class TGNode:
    """
    A node in the transition graph for HMM-based map matching.

    Attributes:
        candidate: The candidate road segment
        prev: Previous optimal node in the path
        ep: Emission probability
        tp: Transition probability from previous optimal candidate
        cumu_prob: Cumulative probability (log-likelihood)
        sp_dist: Shortest path distance from previous optimal candidate
    """

    def __init__(self, candidate: Candidate, ep: float):
        self.candidate = candidate
        self.prev: Optional[TGNode] = None
        self.ep = ep
        self.tp = 0.0
        self.cumu_prob = -float('inf')
        self.sp_dist = 0.0

    def __repr__(self):
        return f"TGNode(edge={self.candidate.edge}, ep={self.ep:.4f}, cumu_prob={self.cumu_prob:.4f})"


class UBODT:
    """
    Upper Bounded Origin Destination Table for precomputed shortest paths.

    This class precomputes and stores shortest paths between all node pairs
    within a specified distance bound (delta). This enables O(1) shortest
    path lookups during map matching.
    """

    def __init__(self, graph: nx.DiGraph, delta: float):
        """
        Initialize UBODT with precomputed shortest paths.

        Args:
            graph: NetworkX directed graph representing road network
            delta: Upper bound distance for shortest path precomputation
        """
        self._logger = getLogger()
        self.graph = graph
        self.delta = delta
        self.table: Dict[Tuple[int, int], Dict] = {}
        self._build_table()

    def _build_table(self):
        """Build the UBODT table using Dijkstra from each source node."""
        self._logger.info(f"Building UBODT with delta={self.delta}...")
        nodes = list(self.graph.nodes())

        for i, source in enumerate(nodes):
            if (i + 1) % 100 == 0:
                self._logger.debug(f"UBODT progress: {i + 1}/{len(nodes)} nodes")

            # Run bounded Dijkstra from source
            distances, predecessors = self._bounded_dijkstra(source)

            for target, cost in distances.items():
                if source != target and cost <= self.delta:
                    # Find first edge on path to target
                    first_n = self._get_first_node_on_path(source, target, predecessors)
                    if first_n is not None:
                        self.table[(source, target)] = {
                            'cost': cost,
                            'first_n': first_n,
                            'prev_n': predecessors.get(target)
                        }

        self._logger.info(f"UBODT built with {len(self.table)} entries")

    def _bounded_dijkstra(self, source) -> Tuple[Dict, Dict]:
        """
        Run Dijkstra's algorithm with upper bound distance.

        Args:
            source: Source node

        Returns:
            Tuple of (distances dict, predecessors dict)
        """
        distances = {source: 0}
        predecessors = {}
        heap = [(0, source)]
        visited = set()

        while heap:
            d, u = heapq.heappop(heap)

            if u in visited:
                continue
            visited.add(u)

            if d > self.delta:
                continue

            for v in self.graph.neighbors(u):
                edge_data = self.graph.get_edge_data(u, v)
                weight = edge_data.get('distance', edge_data.get('weight', 1))
                new_dist = d + weight

                if new_dist <= self.delta and (v not in distances or new_dist < distances[v]):
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return distances, predecessors

    def _get_first_node_on_path(self, source, target, predecessors) -> Optional[int]:
        """Get the first node on the path from source to target."""
        if target == source:
            return None

        # Backtrack to find first node after source
        path = [target]
        current = target
        while current in predecessors and predecessors[current] != source:
            current = predecessors[current]
            path.append(current)

        if current in predecessors and predecessors[current] == source:
            return current
        return None

    def look_up(self, source, target) -> Optional[Dict]:
        """
        Look up shortest path information between source and target.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Dict with 'cost', 'first_n', 'prev_n' if path exists, None otherwise
        """
        return self.table.get((source, target))

    def get_sp_distance(self, source, target) -> float:
        """Get shortest path distance, returns infinity if no path."""
        record = self.look_up(source, target)
        if record is None:
            return float('inf')
        return record['cost']


class FMM(AbstractTraditionModel):
    """
    Fast Map Matching (FMM) Algorithm.

    This implementation follows the original FMM paper which uses:
    1. K-nearest neighbor candidate search with spatial indexing
    2. HMM-based map matching with emission and transition probabilities
    3. Precomputed UBODT for fast shortest path queries
    4. Viterbi algorithm for optimal path inference

    Key Parameters:
        k: Number of candidate edges per GPS point (default: 8)
        r: Search radius for candidate edges in meters (default: 300)
        gps_error: GPS measurement error standard deviation (default: 50)
        delta: Upper bound for UBODT precomputation (default: 3000)
        reverse_tolerance: Allowed proportion of reverse movement (default: 0.0)
    """

    def __init__(self, config, data_feature):
        """
        Initialize FMM model.

        Args:
            config: Configuration dictionary with model parameters
            data_feature: Data feature dictionary with dataset properties
        """
        super().__init__(config, data_feature)

        # Logger
        self._logger = getLogger()

        # FMM algorithm parameters (matching original C++ defaults)
        self.k = config.get('k', 8)  # Number of candidates
        self.r = config.get('r', 300)  # Search radius in meters
        self.gps_error = config.get('gps_error', 50)  # GPS error in meters
        self.delta = config.get('delta', 3000)  # UBODT upper bound
        self.reverse_tolerance = config.get('reverse_tolerance', 0.0)

        # Data parameters
        self.with_time = data_feature.get('with_time', True)
        self.with_rd_speed = data_feature.get('with_rd_speed', True)

        # Use UBODT for fast matching (can be disabled for on-the-fly computation)
        self.use_ubodt = config.get('use_ubodt', True)

        # Precompute UBODT lazily
        self._ubodt: Optional[UBODT] = None

        # Data cache
        self.rd_nwk: Optional[nx.DiGraph] = None
        self.usr_id = None
        self.traj_id = None
        self.trajectory = None
        self.res_dct: Dict = {}

        # Spatial indexing parameters
        self.lon_r = None
        self.lat_r = None

    def run(self, data: Dict) -> Dict:
        """
        Run FMM map matching on trajectories.

        Args:
            data: Dictionary containing:
                - 'rd_nwk': NetworkX graph of road network
                - 'trajectory': Dict of user_id -> traj_id -> trajectory data

        Returns:
            Dictionary of matching results: usr_id -> traj_id -> matched_sequence
        """
        self.rd_nwk = data['rd_nwk']
        trajectory = data['trajectory']

        # Build UBODT if enabled
        if self.use_ubodt and self._ubodt is None:
            self._ubodt = UBODT(self.rd_nwk, self.delta)

        # Set spatial search parameters based on first node
        first_node = list(self.rd_nwk.nodes)[0]
        self._set_lon_lat_radius(
            self.rd_nwk.nodes[first_node]['lon'],
            self.rd_nwk.nodes[first_node]['lat']
        )

        # Process each trajectory
        for usr_id, usr_value in trajectory.items():
            self.usr_id = usr_id
            for traj_id, value in usr_value.items():
                self._logger.info(f'FMM: begin map matching, usr_id:{usr_id} traj_id:{traj_id}')
                self.traj_id = traj_id
                self.trajectory = value
                self._run_one_trajectory()
                self._logger.info(f'FMM: finish map matching, usr_id:{usr_id} traj_id:{traj_id}')

        return self.res_dct

    def _set_lon_lat_radius(self, lon: float, lat: float):
        """
        Compute search radius in degrees for grid-based candidate search.

        Args:
            lon: Reference longitude
            lat: Reference latitude
        """
        # Latitude radius (constant across Earth)
        self.lat_r = radian2angle(self.r / R_EARTH)

        # Longitude radius (varies with latitude)
        r_prime = R_EARTH * math.cos(angle2radian(lat))
        self.lon_r = radian2angle(self.r / r_prime)

    def _run_one_trajectory(self):
        """Run FMM algorithm for a single trajectory."""
        # Step 1: Get candidates for each GPS point
        candidates_per_point = self._get_candidates()
        self._logger.debug(f'Found candidates for {len(candidates_per_point)} points')

        if not candidates_per_point or all(len(c) == 0 for c in candidates_per_point):
            self._logger.warning('No candidates found for trajectory')
            self._store_empty_result()
            return

        # Step 2: Build transition graph with emission probabilities
        tg_layers = self._build_transition_graph(candidates_per_point)
        self._logger.debug('Built transition graph')

        # Step 3: Update transition probabilities using Viterbi-like algorithm
        self._update_transition_graph(tg_layers)
        self._logger.debug('Updated transition probabilities')

        # Step 4: Backtrack to find optimal path
        optimal_path = self._backtrack(tg_layers)
        self._logger.debug(f'Found optimal path with {len(optimal_path)} nodes')

        # Step 5: Store results
        self._store_result(optimal_path)

    def _get_candidates(self) -> List[List[Candidate]]:
        """
        Find k-nearest candidate edges for each GPS point.

        Returns:
            List of candidate lists, one per GPS point
        """
        # Trajectory format: [dyna_id, lon, lat, time, ...]
        traj_lon_lat = self.trajectory[:, 1:3]
        candidates_per_point = []

        for i in range(traj_lon_lat.shape[0]):
            lon, lat = traj_lon_lat[i, :]
            point_candidates = []

            # Find edges within search radius
            for edge in self.rd_nwk.edges:
                source, target = edge[:2]

                src_lat = self.rd_nwk.nodes[source]['lat']
                src_lon = self.rd_nwk.nodes[source]['lon']
                tgt_lat = self.rd_nwk.nodes[target]['lat']
                tgt_lon = self.rd_nwk.nodes[target]['lon']

                # Quick grid-based filter
                if not self._edge_in_radius(lon, lat, src_lon, src_lat, tgt_lon, tgt_lat):
                    continue

                # Compute exact distance and offset
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
                        edge_length=edge_length
                    )
                    point_candidates.append(candidate)

            # Sort by distance and keep top k
            point_candidates.sort(key=lambda c: c.dist)
            candidates_per_point.append(point_candidates[:self.k])

        return candidates_per_point

    def _edge_in_radius(self, lon: float, lat: float,
                        src_lon: float, src_lat: float,
                        tgt_lon: float, tgt_lat: float) -> bool:
        """Check if edge is within search radius using grid filter."""
        # Check if either endpoint or midpoint is in radius
        if (lat - self.lat_r <= src_lat <= lat + self.lat_r and
            lon - self.lon_r <= src_lon <= lon + self.lon_r):
            return True
        if (lat - self.lat_r <= tgt_lat <= lat + self.lat_r and
            lon - self.lon_r <= tgt_lon <= lon + self.lon_r):
            return True

        # Check midpoint
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
        """
        Compute perpendicular distance from point to edge, offset, and projection.

        Uses great circle distance calculations for geographic coordinates.

        Args:
            lon, lat: GPS point coordinates
            src_lon, src_lat: Edge source coordinates
            tgt_lon, tgt_lat: Edge target coordinates

        Returns:
            Tuple of (distance, offset, projected_point)
        """
        # Convert to radians for distance calculation
        lat_r = angle2radian(lat)
        lon_r = angle2radian(lon)
        src_lat_r = angle2radian(src_lat)
        src_lon_r = angle2radian(src_lon)
        tgt_lat_r = angle2radian(tgt_lat)
        tgt_lon_r = angle2radian(tgt_lon)

        # Distances to endpoints
        a = dist(lat_r, lon_r, src_lat_r, src_lon_r)  # Point to source
        b = dist(lat_r, lon_r, tgt_lat_r, tgt_lon_r)  # Point to target
        c = dist(src_lat_r, src_lon_r, tgt_lat_r, tgt_lon_r)  # Edge length

        # Handle degenerate case
        if c == 0:
            return a, 0, (src_lat, src_lon)

        # Check if projection is beyond source (offset < 0)
        if b * b > a * a + c * c:
            return a, 0, (src_lat, src_lon)

        # Check if projection is beyond target (offset > edge_length)
        if a * a > b * b + c * c:
            return b, c, (tgt_lat, tgt_lon)

        # Projection is on the edge - use Heron's formula
        p = (a + b + c) / 2
        area_sq = p * abs(p - a) * abs(p - b) * abs(p - c)
        if area_sq < 0:
            area_sq = 0
        area = math.sqrt(area_sq)
        distance = 2 * area / c

        # Compute offset using Pythagorean theorem
        offset = math.sqrt(max(0, a * a - distance * distance))

        # Compute projected point (linear interpolation for simplicity)
        t = offset / c if c > 0 else 0
        proj_lat = src_lat + t * (tgt_lat - src_lat)
        proj_lon = src_lon + t * (tgt_lon - src_lon)

        return distance, offset, (proj_lat, proj_lon)

    def _build_transition_graph(self, candidates_per_point: List[List[Candidate]]) -> List[List[TGNode]]:
        """
        Build transition graph with emission probabilities.

        Args:
            candidates_per_point: Candidates for each GPS point

        Returns:
            List of TGNode layers
        """
        layers = []

        for candidates in candidates_per_point:
            layer = []
            for candidate in candidates:
                ep = self._calc_emission_prob(candidate.dist)
                node = TGNode(candidate, ep)
                layer.append(node)
            layers.append(layer)

        # Initialize first layer with emission probabilities only
        if layers and layers[0]:
            for node in layers[0]:
                node.cumu_prob = math.log(node.ep) if node.ep > 0 else -float('inf')

        return layers

    def _calc_emission_prob(self, distance: float) -> float:
        """
        Calculate emission probability using Gaussian distribution.

        p(z|r) = (1 / sqrt(2*pi) / sigma) * exp(-0.5 * (d/sigma)^2)

        For numerical stability in log space, we use:
        log(p) proportional to -0.5 * (d/sigma)^2

        Args:
            distance: Distance from GPS point to candidate edge

        Returns:
            Emission probability
        """
        a = distance / self.gps_error
        return math.exp(-0.5 * a * a)

    def _calc_transition_prob(self, sp_dist: float, eu_dist: float) -> float:
        """
        Calculate transition probability.

        Based on FMM paper:
        tp = eu_dist / sp_dist if sp_dist >= eu_dist else 1.0

        This favors paths where network distance matches Euclidean distance.

        Args:
            sp_dist: Shortest path distance between candidates
            eu_dist: Euclidean distance between GPS points

        Returns:
            Transition probability
        """
        if sp_dist == 0:
            return 1.0
        if sp_dist == float('inf'):
            return 0.0
        return min(1.0, eu_dist / sp_dist)

    def _update_transition_graph(self, layers: List[List[TGNode]]):
        """
        Update transition graph using Viterbi-like forward pass.

        For each pair of consecutive layers, compute shortest path distances
        and update cumulative probabilities.
        """
        # Compute Euclidean distances between consecutive GPS points
        traj_lon_lat = self.trajectory[:, 1:3]
        eu_dists = []
        for i in range(len(traj_lon_lat) - 1):
            lon1, lat1 = traj_lon_lat[i]
            lon2, lat2 = traj_lon_lat[i + 1]
            d = dist(angle2radian(lat1), angle2radian(lon1),
                    angle2radian(lat2), angle2radian(lon2))
            eu_dists.append(d)

        # Update each layer pair
        for i in range(len(layers) - 1):
            if not layers[i] or not layers[i + 1]:
                continue

            self._update_layer(layers[i], layers[i + 1], eu_dists[i])

    def _update_layer(self, layer_a: List[TGNode], layer_b: List[TGNode], eu_dist: float):
        """
        Update transition probabilities between two layers.

        Args:
            layer_a: Previous layer
            layer_b: Current layer
            eu_dist: Euclidean distance between corresponding GPS points
        """
        for node_b in layer_b:
            best_prob = -float('inf')
            best_prev = None
            best_tp = 0.0
            best_sp_dist = 0.0

            for node_a in layer_a:
                # Get shortest path distance
                sp_dist = self._get_sp_dist(node_a.candidate, node_b.candidate)

                # Calculate transition probability
                tp = self._calc_transition_prob(sp_dist, eu_dist)

                # Calculate cumulative probability (in log space)
                if node_a.cumu_prob > -float('inf') and tp > 0 and node_b.ep > 0:
                    cumu_prob = node_a.cumu_prob + math.log(tp) + math.log(node_b.ep)
                else:
                    cumu_prob = -float('inf')

                if cumu_prob > best_prob:
                    best_prob = cumu_prob
                    best_prev = node_a
                    best_tp = tp
                    best_sp_dist = sp_dist

            node_b.cumu_prob = best_prob
            node_b.prev = best_prev
            node_b.tp = best_tp
            node_b.sp_dist = best_sp_dist

    def _get_sp_dist(self, ca: Candidate, cb: Candidate) -> float:
        """
        Get shortest path distance between two candidates.

        Handles special cases:
        1. Same edge, forward movement: offset difference
        2. Same edge, small backward movement (within tolerance): 0
        3. Adjacent edges: direct computation
        4. Otherwise: UBODT lookup or on-the-fly Dijkstra

        Args:
            ca: Source candidate
            cb: Target candidate

        Returns:
            Shortest path distance
        """
        source_a, target_a = ca.edge
        source_b, target_b = cb.edge

        # Case 1: Same edge, forward movement
        if ca.edge == cb.edge and ca.offset <= cb.offset:
            return cb.offset - ca.offset

        # Case 2: Same edge, small backward movement (within tolerance)
        if ca.edge == cb.edge:
            backward_dist = ca.offset - cb.offset
            if backward_dist < ca.edge_length * self.reverse_tolerance:
                return 0

        # Case 3: Adjacent edges (target of a is source of b)
        if target_a == source_b:
            return ca.edge_length - ca.offset + cb.offset

        # Case 4: Need shortest path lookup
        if self._ubodt is not None:
            # Use precomputed UBODT
            record = self._ubodt.look_up(target_a, source_b)
            if record is None:
                return float('inf')
            return record['cost'] + ca.edge_length - ca.offset + cb.offset
        else:
            # On-the-fly computation using NetworkX
            try:
                sp_length = nx.shortest_path_length(
                    self.rd_nwk, target_a, source_b, weight='distance'
                )
                return sp_length + ca.edge_length - ca.offset + cb.offset
            except nx.NetworkXNoPath:
                return float('inf')

    def _backtrack(self, layers: List[List[TGNode]]) -> List[TGNode]:
        """
        Backtrack through transition graph to find optimal path.

        Args:
            layers: Transition graph layers

        Returns:
            Optimal path as list of TGNodes
        """
        if not layers:
            return []

        # Find best node in last non-empty layer
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

        # Backtrack
        path = []
        current = best_node
        while current is not None:
            path.append(current)
            current = current.prev

        path.reverse()
        return path

    def _store_result(self, optimal_path: List[TGNode]):
        """Store matching result in the result dictionary."""
        if not optimal_path:
            self._store_empty_result()
            return

        # Build result array
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

    def get_matched_geometry(self, optimal_path: List[TGNode]) -> List[Tuple[float, float]]:
        """
        Get the matched geometry as a list of coordinates.

        Args:
            optimal_path: Optimal path from backtracking

        Returns:
            List of (lat, lon) coordinates
        """
        if not optimal_path:
            return []

        geometry = []
        for node in optimal_path:
            geometry.append(node.candidate.point)

        return geometry
