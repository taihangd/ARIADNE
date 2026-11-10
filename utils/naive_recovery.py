"""Explicit rule-based baseline: interpolates timestamps along precomputed shortest paths only when near-straight and speed-feasible."""

from collections.abc import Mapping
import math
from numbers import Real


def _finite(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def normalize_naive_rules(config=None):
    """Parameters represent pre-declared runtime rules and never read evaluation scores."""
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - {"max_detour_ratio", "max_speed_mps"}:
        raise ValueError("naive_rules accepts only max_detour_ratio and max_speed_mps")
    rules = {"max_detour_ratio": 1.2, "max_speed_mps": 65.0, **config}
    if not _finite(rules["max_detour_ratio"]) or rules["max_detour_ratio"] < 1:
        raise ValueError("max_detour_ratio must be finite and at least 1")
    if not _finite(rules["max_speed_mps"]) or rules["max_speed_mps"] <= 0:
        raise ValueError("max_speed_mps must be finite and positive")
    return {key: float(value) for key, value in rules.items()}


def _xy(graph, node):
    try:
        values = graph.nodes[node]
        xy = values.get("x"), values.get("y")
        return tuple(float(value) for value in xy) if all(_finite(value) for value in xy) else None
    except (KeyError, TypeError):
        return None


def map_recovery_observations(query, *, camera_to_node, road_graph=None):
    """Validate inputs and camera-to-node mappings for pre-API checks without computing paths or reading labels."""
    from utils.fixed_expert_recovery import _query
    _query(query)
    if not isinstance(camera_to_node, Mapping):
        raise ValueError("Recovery baseline requires an explicit camera-to-road-node mapping")
    trace = query.get("camera_trace")
    if not isinstance(trace, (list, tuple)):
        raise ValueError("Recovery baseline requires an explicit camera_trace")
    observed = []
    for item in trace:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not _finite(item[1]):
            raise ValueError("Camera observations require finite timestamps")
        camera, stamp = item
        try:
            node = camera_to_node[camera]
            present = road_graph is None or node in road_graph
        except (KeyError, TypeError):
            raise ValueError("Every observation requires an explicit camera-to-road-node mapping") from None
        if not present:
            raise ValueError("Mapped observation node is absent from the road graph")
        if road_graph is not None and _xy(road_graph, node) is None:
            raise ValueError("Observed road nodes require finite planar coordinates")
        observed.append((node, stamp))
    return observed


def _validated_path(graph, entry, start, end):
    """Use only the cached node sequence and rebuild metric lengths from actual edge weights rather than legacy cached numbers.

    Legacy preprocessing on a MultiDiGraph may store hop counts. The rule baseline treats the road graph
    as authoritative for metric distance, selecting the minimum positive length across parallel edges without modifying legacy shortest-path caches.
    """
    if not isinstance(entry, (list, tuple)) or len(entry) != 3:
        return None, "invalid_cached_path"
    nodes = entry[0]
    if (not isinstance(nodes, (list, tuple)) or len(nodes) < 2
            or nodes[0] != start or nodes[-1] != end):
        return None, "invalid_cached_path"
    lengths = [0.0]
    for index, node in enumerate(nodes):
        if _xy(graph, node) is None:
            return None, "invalid_node_coordinates"
        if index == 0:
            continue
        try:
            edge = graph.get_edge_data(nodes[index - 1], node)
        except TypeError:
            return None, "invalid_cached_path"
        if edge is None:
            return None, "missing_directed_edge"
        edges = list(edge.values()) if graph.is_multigraph() else [edge]
        known = [item.get("length") for item in edges if isinstance(item, Mapping) and "length" in item]
        valid = [float(value) for value in known if _finite(value) and value > 0]
        if not valid:
            return None, "invalid_edge_length" if known else "missing_edge_length"
        weight = min(valid)
        left_xy, right_xy = _xy(graph, nodes[index - 1]), _xy(graph, node)
        chord = math.hypot(right_xy[0] - left_xy[0], right_xy[1] - left_xy[1])
        if (not _finite(chord) or weight < chord
                and not math.isclose(weight, chord, rel_tol=1e-9, abs_tol=1e-6)):
            return None, "invalid_edge_length"
        length = lengths[-1] + weight
        if not _finite(length):
            return None, "invalid_edge_length"
        lengths.append(length)
    return (nodes, lengths, lengths[-1]), None


def _cached_lengths_match(entry, rebuilt):
    """Report whether legacy cached lengths match; rule checks and interpolation always use rebuilt lengths."""
    cumulative, total = entry[1:]
    return (isinstance(cumulative, (list, tuple)) and len(cumulative) == len(rebuilt[1])
            and _finite(total) and all(_finite(value) for value in cumulative)
            and math.isclose(total, rebuilt[2], rel_tol=1e-9, abs_tol=1e-6)
            and all(math.isclose(cached, actual, rel_tol=1e-9, abs_tol=1e-6)
                    for cached, actual in zip(cumulative, rebuilt[1])))


def recover_rule_shortest_path(query, *, road_graph, camera_to_node, shortest_paths, rule_config=None):
    """Return road-node timestamps and per-segment audit records without sorting, dropping points, or inferring target identity.

    Near-straight means the ratio of rebuilt road-network path length to straight-line chord distance does not exceed
    the detour threshold and the implied speed along the path does not exceed max_speed_mps. Unresolved segments retain endpoint observations and record the rejection reason.
    """
    rules = normalize_naive_rules(rule_config)
    if (road_graph is None or not isinstance(camera_to_node, Mapping)
            or not isinstance(shortest_paths, Mapping)):
        raise ValueError("rule_shortest_path requires graph, camera mapping and preprocessed paths")
    observed = map_recovery_observations(query, camera_to_node=camera_to_node, road_graph=road_graph)

    from common import complete_trajectories_fast
    trajectory = observed[:1]
    segments = []
    for index, (start, end) in enumerate(zip(observed, observed[1:])):
        u, t1 = start
        v, t2 = end
        dt = t2 - t1
        audit = {"segment_index": index, "from_node": u, "to_node": v,
                 "start_time": t1, "end_time": t2, "time_delta_sec": dt if _finite(dt) else None,
                 "shortest_path_length_m": None, "straight_line_distance_m": None,
                 "detour_ratio": None, "implied_speed_mps": None, "interpolated_count": 0,
                 "distance_source": None, "lengths_rebuilt": False, "legacy_cached_lengths_match": None}
        segments.append(audit)
        if not _finite(dt):
            audit["reason"] = "nonfinite_time_delta"
        elif dt <= 0:
            audit["reason"] = "reverse_time" if dt < 0 else "zero_time_delta"
        elif u == v:
            audit.update(reason="same_node", shortest_path_length_m=0.0,
                         straight_line_distance_m=0.0, implied_speed_mps=0.0)
        else:
            xy1, xy2 = _xy(road_graph, u), _xy(road_graph, v)
            if xy1 is None or xy2 is None:
                audit["reason"] = "invalid_node_coordinates"
            else:
                chord = math.hypot(xy2[0] - xy1[0], xy2[1] - xy1[1])
                audit["straight_line_distance_m"] = chord if _finite(chord) else None
                entry = shortest_paths.get((u, v))
                path, error = _validated_path(road_graph, entry, u, v) if entry is not None else (
                    None, "missing_shortest_path")
                if error:
                    audit["reason"] = error
                else:
                    length = float(path[2])
                    speed = length / dt
                    audit.update(shortest_path_length_m=length,
                                 implied_speed_mps=speed if _finite(speed) else None,
                                 distance_source="road_graph_edge_lengths", lengths_rebuilt=True,
                                 legacy_cached_lengths_match=_cached_lengths_match(entry, path))
                    if not _finite(chord):
                        audit["reason"] = "nonfinite_chord_distance"
                    elif chord == 0:
                        audit["reason"] = "coincident_distinct_nodes"
                    elif length < chord and not math.isclose(length, chord, rel_tol=1e-9, abs_tol=1e-6):
                        audit["reason"] = "path_length_below_chord_distance"
                    else:
                        ratio = length / chord
                        audit["detour_ratio"] = ratio if _finite(ratio) else None
                        if not _finite(ratio):
                            audit["reason"] = "nonfinite_detour_ratio"
                        elif ratio > rules["max_detour_ratio"] and not math.isclose(
                                ratio, rules["max_detour_ratio"], rel_tol=1e-12):
                            audit["reason"] = "excessive_detour"
                        elif not _finite(speed):
                            audit["reason"] = "nonfinite_implied_speed"
                        elif speed > rules["max_speed_mps"] and not math.isclose(
                                speed, rules["max_speed_mps"], rel_tol=1e-12):
                            audit["reason"] = "excessive_speed"
                        else:
                            completed = complete_trajectories_fast({0: [start, end]}, {(u, v): path})[0]
                            trajectory.extend(completed[1:-1])
                            audit.update(reason="within_rules", interpolated_count=len(completed) - 2)
        audit["resolved"] = audit["reason"] in {"within_rules", "same_node"}
        trajectory.append(end)
    unresolved = sum(not segment["resolved"] for segment in segments)
    return {"trajectory": trajectory, "segments": segments, "rule_config": rules,
            "unresolved_segment_count": unresolved,
            "fully_interpolated": bool(observed) and unresolved == 0}
