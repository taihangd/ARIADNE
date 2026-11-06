"""Core road-graph path preprocessing and trajectory point utilities."""

from collections import defaultdict
import numpy as np


def preprocess_paths_with_length(shortest_path_results, road_graph):
    """Attach cumulative edge lengths and total distance to precomputed shortest paths."""
    processed_paths = {}
    for key, paths in shortest_path_results.items():
        path_nodes = paths[0]
        cumulative_lengths = [0.0]
        total_length = 0.0
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = road_graph[u][v]
            if "length" in edge_data:
                length = float(edge_data["length"])
            elif edge_data:
                length = float(min(
                    (d.get("length", 1.0) for d in edge_data.values() if isinstance(d, dict)),
                    default=1.0,
                ))
            else:
                length = 1.0
            total_length += length
            cumulative_lengths.append(total_length)
        processed_paths[key] = (path_nodes, cumulative_lengths, total_length)
    return processed_paths


def complete_trajectories_fast(cluster_traj_dict, processed_paths):
    """Interpolate along precomputed shortest paths while preserving observed endpoints."""
    completed = {}
    for label, observations in cluster_traj_dict.items():
        trajectory = [tuple(observations[0])] if observations else []
        for start, end in zip(observations, observations[1:]):
            rid1, t1 = start
            rid2, t2 = end
            entry = processed_paths.get((rid1, rid2))
            if entry is not None:
                nodes, lengths, total = entry
                if (nodes and nodes[0] == rid1 and nodes[-1] == rid2
                        and len(nodes) == len(lengths) and total > 0):
                    trajectory.extend(
                        (node, t1 + distance / total * (t2 - t1))
                        for node, distance in zip(nodes[1:-1], lengths[1:-1])
                    )
            trajectory.append((rid2, t2))
        completed[label] = trajectory
    return completed


def merge_tm_adj_points(points, adj_range=100):
    """Merge temporally adjacent trajectory points at the same node within adj_range seconds."""
    node_to_tms = defaultdict(list)
    for node, tm in points:
        node_to_tms[node].append(tm)

    merge_points = []
    for node, tms in node_to_tms.items():
        if len(tms) == 1:
            merge_points.append((node, tms[0]))
            continue
        tms.sort()
        min_tm = tms[0]
        one_cluster = [tms[0]]
        for tm in tms[1:]:
            if tm - min_tm <= adj_range:
                one_cluster.append(tm)
            else:
                merge_points.append((node, float(np.mean(one_cluster))))
                one_cluster = [tm]
                min_tm = tm
        merge_points.append((node, float(np.mean(one_cluster))))
    return merge_points
