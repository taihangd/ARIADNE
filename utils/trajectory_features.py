"""Authoritative observation feature implementation shared across retrieval and API routing.

Coordinates are used only when units are explicitly declared. Temporal statistics preserve
the caller's unit; the >30s gap proxy feature is computed only when seconds are explicitly declared.
Graph statistics use mapped observations and shortest-path hop counts without fabricating paths from camera IDs.
This module has no model initialization or data loading side effects."""
import math
import numpy as np

OBSERVATION_SCHEMA = (
    "obs_len", "time_span", "transition_count", "time_gap", "time_gap_cv",
    "shortest_path_mean", "shortest_path_max", "unreachable_ratio", "out_degree_mean",
)
FEATURE_SCHEMA = OBSERVATION_SCHEMA
FEATURE_VERSION = "observations-v2-explicit-units"


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def build_observation_features(camera_trace, road_graph=None, camera_to_node=None,
                               cam_xy_dict=None, rid_trace=None, *, time_unit="caller",
                               coordinate_unit="unknown"):
    """Build pre-call features while preserving observation order; features remain None when evidence or units are missing."""
    trace = list(camera_trace or [])
    cameras = [camera for camera, _ in trace]
    times = [_finite(timestamp) for _, timestamp in trace]
    result = dict.fromkeys(OBSERVATION_SCHEMA)
    result.update(feature_schema=list(OBSERVATION_SCHEMA), feature_version=FEATURE_VERSION,
                  time_unit=time_unit, coordinate_unit=coordinate_unit, obs_len=len(trace),
                  transition_count=sum(a != b for a, b in zip(cameras, cameras[1:])),
                  traj_len=float(len(trace)), spatial_range=None, jump_distance=None,
                  branching_score=None, missing_ratio=None)
    if times and all(value is not None for value in times):
        result["time_span"] = max(times) - min(times)
        gaps = [b - a for a, b in zip(times, times[1:])]
        if gaps:
            mean = sum(gaps) / len(gaps)
            variance = sum((value - mean) ** 2 for value in gaps) / len(gaps)
            result["time_gap"] = mean
            result["time_gap_cv"] = math.sqrt(variance) / abs(mean) if mean else (0. if not any(gaps) else None)
            if time_unit in {"s", "sec", "second", "seconds"}:
                result["missing_ratio"] = sum(value > 30 for value in gaps) / len(gaps)
    # Legacy branching_score serves as an observation uniqueness proxy.
    road_ids = [node for node, _ in rid_trace] if rid_trace else cameras
    if cameras:
        result["branching_score"] = (len(set(cameras)) / len(cameras) +
                                      len(set(road_ids)) / len(road_ids)) / 2
    if coordinate_unit in {"m", "meter", "metre", "meters", "metres"} and cameras:
        points = [(cam_xy_dict or {}).get(camera) for camera in cameras]
        coords = []
        for point in points:
            if point is None or not isinstance(point, (tuple, list)) or len(point) < 2:
                break
            xy = (_finite(point[0]), _finite(point[1]))
            if None in xy:
                break
            coords.append(xy)
        if len(coords) == len(cameras):
            xs, ys = zip(*coords)
            result["spatial_range"] = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            jumps = [math.dist(a, b) for a, b in zip(coords, coords[1:])]
            result["jump_distance"] = sum(jumps) / len(jumps) if jumps else 0.
    if road_graph is not None and camera_to_node is not None:
        import networkx as nx
        nodes = [camera_to_node.get(camera) for camera in cameras]
        nodes = [node if node is not None and node in road_graph else None for node in nodes]
        mapped = [node for node in nodes if node is not None]
        if mapped:
            degree = road_graph.out_degree if road_graph.is_directed() else road_graph.degree
            result["out_degree_mean"] = sum(degree(node) for node in mapped) / len(mapped)
        pairs = [(a, b) for a, b in zip(nodes, nodes[1:]) if a is not None and b is not None]
        hops = []
        for source, target in pairs:
            try:
                hops.append(nx.shortest_path_length(road_graph, source, target))
            except nx.NetworkXNoPath:
                pass
        if pairs:
            result["unreachable_ratio"] = 1 - len(hops) / len(pairs)
        if hops:
            result["shortest_path_mean"] = sum(hops) / len(hops)
            result["shortest_path_max"] = max(hops)
    return result


def build_signature(camera_trace, road_graph=None, camera_to_node=None):
    """Project shared observation-only features onto the 9-feature structural schema."""
    base = build_observation_features(
        camera_trace=camera_trace, road_graph=road_graph, camera_to_node=camera_to_node
    )
    return dict(feature_schema=list(FEATURE_SCHEMA), **{field: base[field] for field in FEATURE_SCHEMA})


def _check_schema(signature, required=False):
    declared = signature.get("feature_schema")
    if (required and declared is None) or (declared is not None and tuple(declared) != FEATURE_SCHEMA):
        raise ValueError("signature feature_schema does not match the fixed model")


def _values(signature):
    _check_schema(signature)
    values = np.zeros(len(FEATURE_SCHEMA), dtype=float)
    valid = np.zeros(len(FEATURE_SCHEMA), dtype=bool)
    for index, name in enumerate(FEATURE_SCHEMA):
        val = _finite(signature.get(name))
        if val is not None:
            values[index], valid[index] = float(val), True
    return values, valid


class FeatureScaler:
    """Fit per-feature mean and standard deviation using only finite values observed in the training set."""

    def fit(self, signatures):
        rows = [_values(signature) for signature in signatures]
        if not rows:
            raise ValueError("scaler needs nonempty training signatures")
        values, valid = map(np.stack, zip(*rows))
        self.count = valid.sum(axis=0)
        self.mean = np.divide(
            values.sum(axis=0), self.count, out=np.zeros(len(FEATURE_SCHEMA)), where=self.count > 0
        )
        variance = np.divide(
            (((values - self.mean) * valid) ** 2).sum(axis=0),
            self.count,
            out=np.zeros(len(FEATURE_SCHEMA)),
            where=self.count > 0,
        )
        self.scale = np.sqrt(variance)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, signature):
        if not hasattr(self, "count"):
            raise RuntimeError("fit the scaler on training data first")
        values, valid = _values(signature)
        valid &= self.count > 0
        return np.where(valid, (values - self.mean) / self.scale, 0.0), valid

    def to_dict(self):
        return {
            "feature_schema": list(FEATURE_SCHEMA),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        _check_schema(data, required=True)
        obj = cls()
        obj.mean = np.asarray(data["mean"], dtype=float)
        obj.scale = np.asarray(data["scale"], dtype=float)
        counts = np.asarray(data["count"], dtype=float)
        if any(
            array.shape != (len(FEATURE_SCHEMA),) or not np.isfinite(array).all()
            for array in (obj.mean, obj.scale, counts)
        ):
            raise ValueError("invalid scaler dimensions or values")
        if np.any(obj.scale <= 0) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("invalid scaler scale/count")
        obj.count = counts.astype(int)
        return obj


def structural_similarity(query_signature, case_signature, scaler, fallback="none"):
    """Compute mean absolute distance across mutually valid scaled features and convert to a monotonic similarity score."""
    if fallback not in {"none", "zero"}:
        raise ValueError("fallback must be none or zero")
    query, query_mask = scaler.transform(query_signature)
    case, case_mask = scaler.transform(case_signature)
    common = query_mask & case_mask
    if not common.any():
        return 0.0 if fallback == "zero" else None
    return float(1.0 / (1.0 + np.mean(np.abs(query[common] - case[common]))))

