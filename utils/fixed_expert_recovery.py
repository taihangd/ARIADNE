"""Core C1 cluster invocation gating, C2 camera-approach prompt/three-tier verification, and C3 corridor subgraph pruning.

Keeps module imports lightweight by loading graph helpers lazily at call time;
ground-truth labels and retrieval scores are never injected into prompts."""

from copy import deepcopy
import json
import math
import numbers
import re


_QUERY_FIELDS = {"query_id", "source_group_id", "camera_trace", "observed_points", "trusted_observation_indices",
                 "required_time_range", "required_time_range_source"}
_CAMERA_MAPPING_CONTEXT_FIELDS = {
    "schema", "observations", "trajectory_scope", "task_scope", "event_order", "pair_evidence", "graph_issues",
    "assumptions", "output_contract", "planning_protocol", "road_intent_settings", "road_context",
    "road_intent_semantics", "camera_mapping_protocol", "camera_mapping_candidates",
    "camera_mapping_coverage_status", "camera_mapping_semantics", "pair_evidence_status", "pair_evidence_clear_reason",
}
_CAMERA_MAPPING_OBSERVATION_FIELDS = {
    "obs_id", "record_index", "input_index", "camera_id", "time", "raw_time_s", "time_conversion",
    "camera_xy", "camera_position_status", "prepared_observation_xy", "measurement_semantics",
    "road_node", "road_node_xy", "mapping_status", "identity_status", "time_reliability", "time_semantics",
}
_CAMERA_MAPPING_PRIVATE_FIELDS = {
    "ground_truth", "ground_truth_by_vehicle", "gt", "vehicle_id", "vehicle_ids", "record_labels",
    "recovery_eval", "evaluation", "f1", "precision", "recall",
}


def _query(query):
    if not isinstance(query, dict) or set(query) - _QUERY_FIELDS:
        raise ValueError("Recovery query accepts only identity and observed trajectory fields; GT is forbidden")
    required_range = query.get("required_time_range")
    if required_range is not None:
        if (not isinstance(required_range, (list, tuple)) or len(required_range) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in required_range)
                or required_range[0] > required_range[1]):
            raise ValueError("required_time_range must be finite [start, end]")
        source = query.get("required_time_range_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("required_time_range needs a declared deployment source; do not infer it from GT")
    return query


def _camera_mapping_public_context(context):
    """Accept only frozen public fields in inference inputs, rejecting any embedded reporting or identity labels prior to deep copying."""
    def public_fields(value):
        if isinstance(value, dict):
            if set(value) & _CAMERA_MAPPING_PRIVATE_FIELDS:
                raise ValueError("Camera mapping input forbids private identity, GT and evaluation fields")
            for item in value.values():
                public_fields(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                public_fields(item)

    if not isinstance(context, dict) or set(context) - _CAMERA_MAPPING_CONTEXT_FIELDS:
        raise ValueError("Camera mapping context accepts only frozen public observation/road fields")
    public_fields(context)
    for obs in context.get("observations", []):
        if not isinstance(obs, dict) or set(obs) - _CAMERA_MAPPING_OBSERVATION_FIELDS:
            raise ValueError("Camera mapping accepts only frozen public observation fields")


def _camera_mapping_graph(context, road_graph):
    """Adapt only a frozen road-intent context while reusing the original graph normalization and digest."""
    from utils.road_time import prepare_road_time_graph

    _camera_mapping_public_context(context)
    if (not isinstance(context, dict) or context.get("schema") != "cluster_tracks_v1"
            or context.get("trajectory_scope") != "single_cluster"
            or context.get("planning_protocol") != "road_intents_v1"
            or not isinstance(context.get("road_intent_settings"), dict)
            or type(context["road_intent_settings"].get("max_waypoints")) is not int
            or context["road_intent_settings"]["max_waypoints"] != 2):
        raise ValueError("Camera mapping requires frozen single_cluster road_intents_v1 with two waypoints")
    prepared = prepare_road_time_graph(road_graph)
    graph = prepared.get("graph")
    if graph is None or prepared.get("hash") != context.get("road_context", {}).get("graph_hash"):
        raise ValueError("Camera mapping requires the unchanged supplied execution graph")
    if any(isinstance(node, bool) or not isinstance(node, numbers.Integral) for node in graph):
        raise ValueError("Camera mapping experiment supports integer road-node IDs only")
    public = context["road_context"].get("nodes", [])
    if (any(not isinstance(row, list) or not row or type(row[0]) is not int for row in public)
            or len(public) != len(graph) or {row[0] for row in public} != set(graph)):
        raise ValueError("Camera mapping requires complete supplied integer road-node coverage")
    return graph


def build_camera_mapping_context(context, *, road_graph, _prepared_graph=None):
    """Copy the public context and attach geometric Top-2 road-node candidates without claiming physical coverage guarantees."""
    graph = _prepared_graph if _prepared_graph is not None else _camera_mapping_graph(context, road_graph)
    if len(graph) < 2:
        raise ValueError("Camera mapping requires at least two supplied road nodes")
    observations = context.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Camera mapping requires public observations")
    coordinates = {}
    for obs in observations:
        if not isinstance(obs, dict) or set(obs) - _CAMERA_MAPPING_OBSERVATION_FIELDS:
            raise ValueError("Camera mapping requires structured public observations")
        camera, xy = obs.get("camera_id"), obs.get("camera_xy")
        if (type(camera) is not int or not isinstance(xy, (list, tuple)) or len(xy) != 2
                or any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                       or not math.isfinite(value) for value in xy)):
            raise ValueError("Camera mapping requires integer camera IDs and finite public camera coordinates")
        xy = tuple(float(value) for value in xy)
        if camera in coordinates and coordinates[camera] != xy:
            raise ValueError("One camera must have consistent public coordinates within the query")
        coordinates[camera] = xy
    copied = deepcopy(context)
    node_xy = [(int(node), graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in graph]
    candidates = []
    for camera in sorted(coordinates):
        xy = coordinates[camera]
        ranked = sorted((math.dist(xy, (nx_val, ny_val)), node)
                        for node, nx_val, ny_val in node_xy)[:2]
        if any(not math.isfinite(distance) for distance, _ in ranked):
            raise ValueError("Camera projection distances must be finite")
        candidates.append({"camera_id": camera, "candidates": [
            {"node": node, "xy": [graph.nodes[node]["x"], graph.nodes[node]["y"]],
             "projection_distance_m": distance} for distance, node in ranked]})
    copied.update(camera_mapping_protocol="camera_mapping_v1", camera_mapping_candidates=candidates,
                  camera_mapping_coverage_status="unknown",
                  camera_mapping_semantics="One conditional supplied Top2 node per camera throughout this query; "
                                           "camera location is not vehicle position. Top2 is a computational limit.",
                  pair_evidence=[], pair_evidence_status="unknown",
                  pair_evidence_clear_reason="Original cached routes depend on the prior fixed mapping; "
                                             "they are not reused under camera_mapping_v1.")
    copied["output_contract"]["camera_node_choices"] = [
        {"camera_id": row["camera_id"], "road_node": "<one supplied candidate integer ID>"}
        for row in candidates]
    return copied


def build_camera_mapping_pair_evidence(context, *, road_graph):
    """Enumerate all candidate endpoint combinations for adjacent observations; distances and implied speeds are provided for comparison without pruning or physical certification."""
    import networkx as nx

    if not isinstance(context, dict) or context.get("camera_mapping_protocol") != "camera_mapping_v1":
        raise ValueError("Pair evidence requires the explicit camera mapping context")
    graph = _camera_mapping_graph(context, road_graph)
    expected = build_camera_mapping_context(context, road_graph=road_graph, _prepared_graph=graph)
    if context.get("camera_mapping_candidates") != expected["camera_mapping_candidates"]:
        raise ValueError("Camera mapping candidates changed from the frozen public geometric Top2")
    candidates = {row["camera_id"]: [item["node"] for item in row["candidates"]]
                  for row in context["camera_mapping_candidates"]}
    distances = {}
    pairs = []
    for left, right in zip(context["observations"], context["observations"][1:]):
        if any(isinstance(obs.get("time"), bool) or not isinstance(obs.get("time"), numbers.Real)
               or not math.isfinite(obs["time"]) for obs in (left, right)):
            raise ValueError("Pair evidence requires finite public source times")
        delta = right["time"] - left["time"]
        if not math.isfinite(delta):
            raise ValueError("Pair evidence requires a finite public time difference")
        alternatives = []
        for source in candidates[left["camera_id"]]:
            if source not in distances:
                distances[source] = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
            for target in candidates[right["camera_id"]]:
                distance = distances[source].get(target)
                consistent = left["camera_id"] != right["camera_id"] or source == target
                speed = distance / delta if consistent and distance is not None and delta > 0 else None
                if speed is not None and not math.isfinite(speed):
                    speed = None
                alternatives.append({
                    "from_node": source, "to_node": target, "same_camera_consistent": consistent,
                    "directed_route_exists_in_supplied_graph": distance is not None,
                    "shortest_distance_m": distance,
                    "required_average_speed_mps_diagnostic": speed,
                    "speed_is_reliable_hard_constraint": False, "physical_feasibility": "unknown",
                })
        pairs.append({"from_obs_id": left["obs_id"], "to_obs_id": right["obs_id"],
                      "observed_time_delta_s": delta, "time_reliability": "unknown",
                      "alternatives": alternatives})
    return {"schema": "camera_mapping_pair_evidence_v1", "candidate_coverage": "unknown",
            "graph_coverage": "unknown", "hard_physical_pruning_applied": False, "pairs": pairs}


def format_mapping_experience_context(payload):
    """Render a complete conditional mapping record; default C2 prompts stay unchanged.

    Retrieval measures this entire block, including the header, before returning
    a payload. No historical absolute times or full road-node paths are retained.
    """
    from utils.fixed_expert_core import _process_public

    if payload is None:
        return ""
    _process_public(payload)
    if (not isinstance(payload, dict) or payload.get("schema") != "conditional_train_mapping_experience_v1"
            or set(payload) != {"schema", "semantics", "camera_support_columns", "node_support_columns",
                                "camera_support", "pair_support_columns", "pair_support", "teacher"}):
        raise ValueError("Render only a complete conditional TRAIN mapping experience payload")
    return ("\n[MAPPING_EXPERIENCE_JSON]\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def build_camera_mapping_prompt(
    context,
    *,
    include_pair_evidence=False,
    road_graph=None,
    retrieved_experience=None,
):
    """Reuse the frozen road-intent prompt while wrapping it with an outer camera-to-node mapping selection schema."""
    from traj_rec_st_cot import build_cluster_track_prompt

    _camera_mapping_public_context(context)
    if context.get("camera_mapping_protocol") != "camera_mapping_v1":
        raise ValueError("Build the explicit camera mapping context before its prompt")
    prompt = build_cluster_track_prompt(context, "evidence")
    old_root = "exactly tracks:[{obs_ids:[...],"
    new_root = "exactly camera_node_choices:[{camera_id:...,road_node:...}],tracks:[{obs_ids:[...],"
    if prompt.count(old_root) != 2:
        raise ValueError("Frozen road-intent output contract changed; review mapping wrapper")
    prompt = prompt.replace(old_root, new_root)
    prompt = ("This is the separate camera_mapping_v1 experiment. Select exactly one integer road_node from "
            "each camera's supplied camera_mapping_candidates. camera_node_choices must cover every supplied "
            "camera exactly once, including cameras of events listed in unassigned_obs_ids. Use one mapping "
            "consistently for repeated observations of that camera and neighboring intervals. Existing "
            "observation road_node fields are the prior conditional mapping, not binding choices. Your chosen "
            "mapping is supplied to the unchanged directed-road executor. Do not change obs_id, camera_id, "
            "coordinates, source timestamps, road graph or length/time rules. Candidate coverage and true "
            "identity/physical validity remain unknown. The complete root object has exactly "
            "camera_node_choices, tracks and unassigned_obs_ids; no extra keys. " + prompt)
    if type(include_pair_evidence) is not bool:
        raise ValueError("include_pair_evidence must be an explicit boolean")
    if include_pair_evidence:
        if road_graph is None:
            raise ValueError("Explicit pair evidence requires the unchanged supplied road graph")
        evidence = build_camera_mapping_pair_evidence(context, road_graph=road_graph)
        prompt = (prompt + "\n[PAIR_COMPARISON_FACTS]\n"
                + json.dumps(evidence, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n[PAIR_COMPARISON_SEMANTICS]\n"
                "These are all public Top2 endpoint combinations for adjacent source observations, "
                "not complete route candidates or a physical feasible-domain guarantee. Directed reachability "
                "and distance refer only to the supplied graph. A missing supplied-graph route does not prove "
                "global physical impossibility. Observed time deltas and required average speeds are diagnostics; "
                "true passage times and reliable maximum speed remain unknown. Do not turn these speeds or "
                "distances into hard acceptance thresholds or assume that the shortest route is the true route. "
                "Same-camera different-node combinations violate the existing one-mapping-per-camera protocol; "
                "all combinations are still displayed. Pair choices must be consistent throughout the query. "
                "Zero distance for equal endpoints permits staying but does not identify whether a loop occurred. "
                "Only original adjacent pairs are tabulated; exclusions may create untabulated intervals. "
                "The full supplied graph and original road-intent actions remain available. "
                "This block does not change the required output object or execution constraints.\n")
    if retrieved_experience is not None:
        block = (
            retrieved_experience
            if isinstance(retrieved_experience, str)
            else format_mapping_experience_context(retrieved_experience)
        )
        if len(block) > 4000:
            raise ValueError("Retrieved mapping experience exceeds 4000-character budget")
        prompt = prompt + block
    return prompt


def parse_camera_mapping_response(text, context, *, road_graph):
    """Parse the strict outer JSON wrapper; camera mappings must be complete and within the candidate set without silent fallbacks, while road intents are validated by the original parser."""
    from traj_rec_st_cot import parse_cluster_track_groups
    from utils.cluster_road_intents import _strip_loose_markdown_fence

    if context.get("camera_mapping_protocol") != "camera_mapping_v1":
        raise ValueError("Camera mapping response requires explicit experiment context")
    expected = build_camera_mapping_context(context, road_graph=road_graph)
    if context.get("camera_mapping_candidates") != expected["camera_mapping_candidates"]:
        raise ValueError("Camera mapping candidates changed from the frozen public geometric Top2")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON keys are forbidden")
            obj[key] = value
        return obj

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constants are forbidden")

    if not isinstance(text, str):
        raise ValueError("Camera mapping requires one complete pure JSON string")
    stripped = text.strip()
    if re.fullmatch(r"```(?:json)?\s*[\s\S]*?\s*```", stripped, flags=re.IGNORECASE):
        stripped = _strip_loose_markdown_fence(stripped)
    payload = json.loads(stripped, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if not isinstance(payload, dict) or set(payload) != {"camera_node_choices", "tracks", "unassigned_obs_ids"}:
        raise ValueError("Camera mapping response has exactly camera_node_choices, tracks and unassigned_obs_ids")
    choices = payload["camera_node_choices"]
    allowed = {row["camera_id"]: {candidate["node"]: candidate for candidate in row["candidates"]}
               for row in expected["camera_mapping_candidates"]}
    if not isinstance(choices, list) or len(choices) != len(allowed):
        raise ValueError("Every supplied camera requires exactly one mapping choice")
    mapping = {}
    for choice in choices:
        if (not isinstance(choice, dict) or set(choice) != {"camera_id", "road_node"}
                or type(choice["camera_id"]) is not int or type(choice["road_node"]) is not int):
            raise ValueError("Mapping choices require exactly integer camera_id and road_node")
        camera, node = choice["camera_id"], choice["road_node"]
        if camera in mapping or camera not in allowed or node not in allowed[camera]:
            raise ValueError("Mapping camera must be unique and its road node a supplied candidate")
        mapping[camera] = node
    if set(mapping) != set(allowed):
        raise ValueError("Mapping choices must cover every supplied camera")
    copied = deepcopy(context)
    copied["pair_evidence"] = []
    copied["pair_evidence_status"] = "unknown"
    copied["pair_evidence_clear_reason"] = expected["pair_evidence_clear_reason"]
    for obs in copied["observations"]:
        node = mapping[obs["camera_id"]]
        obs.update(road_node=node, road_node_xy=deepcopy(allowed[obs["camera_id"]][node]["xy"]),
                   mapping_status="conditional")
    groups = parse_cluster_track_groups(json.dumps({key: payload[key] for key in
                                        ("tracks", "unassigned_obs_ids")}, allow_nan=False), copied)
    return groups, copied, mapping


def build_cluster_corridor_subgraph(
    context,
    *,
    road_graph,
    extra_nodes=(),
    sp_cache=None,
    pair_scope="all_forward",
    camera_to_node=None,
    camera_candidates=None,
    shortest_paths=None,
    k_hop=0,
):
    """Construct the local corridor subgraph G_q subset G for a cluster query (Section III-C2, Eq. 8).

    Args:
        context: Either a cluster track context dict or a raw camera_trace list [(cam_id, ts), ...].
        road_graph: Directed road network graph.
        extra_nodes: Optional iterable of extra road-node IDs (e.g., via_nodes) to include.
        sp_cache: Optional dict caching shortest-path node sequences.
        pair_scope: "consecutive_only" (I_q = {(i, i+1)}), "consecutive_and_two_hop"
            (I_q = {(i, i+1)} U {(i, i+2)} as defined in Eq. 8), or "all_forward" (all j > i).
        camera_to_node: Optional mapping {camera_id: road_node_id} when context is a camera_trace.
        camera_candidates: Optional mapping {camera_id: [cand_node_1, cand_node_2]} (Top-2 candidates Gamma_2(c_i)).
        shortest_paths: Optional precomputed shortest-path dict {(u, v): (nodes, cum_lengths, total_len)}.
        k_hop: Optional neighborhood expansion radius around corridor nodes (default 0; 1 adds N_1).
    """
    import networkx as nx
    from utils.road_time import prepare_road_time_graph

    if road_graph is None:
        raise ValueError("Corridor subgraph construction requires road_graph")
    if pair_scope not in {"consecutive_only", "consecutive_and_two_hop", "all_forward"}:
        raise ValueError("pair_scope must be 'consecutive_only', 'consecutive_and_two_hop', or 'all_forward'")
    sp_cache = sp_cache if sp_cache is not None else {}
    sp_precomputed = shortest_paths if isinstance(shortest_paths, dict) else {}
    norm_graph = None

    def _sp_nodes(src, dst):
        if src == dst:
            return (src,)
        if (src, dst) in sp_cache:
            return sp_cache[(src, dst)]
        if sp_precomputed:
            pre = sp_precomputed.get((src, dst))
            if pre is None and src in sp_precomputed and isinstance(sp_precomputed[src], dict):
                pre = sp_precomputed[src].get(dst)
            if isinstance(pre, (list, tuple)) and len(pre) >= 1 and isinstance(pre[0], (list, tuple)):
                sp_cache[(src, dst)] = tuple(pre[0])
            elif isinstance(pre, (list, tuple)):
                sp_cache[(src, dst)] = tuple(pre)
            else:
                sp_cache[(src, dst)] = ()
            return sp_cache[(src, dst)]
        nonlocal norm_graph
        if norm_graph is None:
            norm_graph = sp_cache.setdefault(
                "_norm_graph", prepare_road_time_graph(road_graph).get("graph") or road_graph
            )
        weight_key = "weight" if norm_graph is not road_graph else "length"
        try:
            sp_cache[(src, dst)] = tuple(nx.shortest_path(norm_graph, src, dst, weight=weight_key))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sp_cache[(src, dst)] = ()
        return sp_cache[(src, dst)]

    if isinstance(context, dict):
        cands = {
            r["camera_id"]: [int(c["node"]) for c in r.get("candidates", []) if type(c.get("node")) is int]
            for r in context.get("camera_mapping_candidates", [])
            if isinstance(r, dict) and type(r.get("camera_id")) is int
        }
        observations = [o for o in context.get("observations", []) if isinstance(o, dict)]
        if not observations and isinstance(context.get("camera_trace"), (list, tuple)) and isinstance(camera_to_node, dict):
            observations = [
                {"camera_id": int(cam), "road_node": int(camera_to_node[cam])}
                for cam, _ in context["camera_trace"]
                if cam in camera_to_node
            ]
    elif isinstance(context, (list, tuple)):
        cands = {
            int(cam): [int(n) for n in ns]
            for cam, ns in (camera_candidates or {}).items()
            if ns
        }
        mapping = camera_to_node if isinstance(camera_to_node, dict) else {}
        observations = [
            {"camera_id": int(cam), "road_node": int(mapping[cam])}
            for cam, _ in context
            if cam in mapping
        ]
    else:
        raise ValueError("context must be a cluster context dict or camera_trace sequence")

    extra = {int(n) for n in (extra_nodes or ()) if isinstance(n, numbers.Integral) and not isinstance(n, bool)}
    nodes = (
        set(extra)
        | {n for ns in cands.values() for n in ns}
        | {o["road_node"] for o in observations if type(o.get("road_node")) is int}
    )
    obs_nodes = [
        cands.get(o.get("camera_id")) or ([o["road_node"]] if "road_node" in o else [])
        for o in observations
    ]
    for i, nodes_a in enumerate(obs_nodes):
        if pair_scope == "consecutive_only":
            target_slice = obs_nodes[i + 1 : i + 2]
        elif pair_scope == "consecutive_and_two_hop":
            target_slice = obs_nodes[i + 1 : i + 3]
        else:
            target_slice = obs_nodes[i + 1 :]
        for ua in nodes_a:
            for v in extra:
                nodes.update(_sp_nodes(ua, v))
                nodes.update(_sp_nodes(v, ua))
            for nodes_b in target_slice:
                for ub in nodes_b:
                    nodes.update(_sp_nodes(ua, ub))
    for v1 in extra:
        for v2 in extra:
            if v1 != v2:
                nodes.update(_sp_nodes(v1, v2))
    if k_hop and int(k_hop) > 0:
        frontier = set(nodes)
        for _ in range(int(k_hop)):
            next_frontier = set()
            for u in frontier:
                if road_graph.has_node(u):
                    next_frontier.update(road_graph.neighbors(u))
                    if road_graph.is_directed() and hasattr(road_graph, "predecessors"):
                        next_frontier.update(road_graph.predecessors(u))
            frontier = next_frontier - nodes
            nodes.update(frontier)
    valid_nodes = [n for n in sorted(nodes) if road_graph.has_node(n)]
    valid_set = set(valid_nodes)
    subgraph = road_graph.__class__()
    for u in valid_nodes:
        subgraph.add_node(u, **road_graph.nodes[u])
    is_multi = road_graph.is_multigraph()
    ec = 0
    for u in valid_nodes:
        for v, datadict in road_graph[u].items():
            if v in valid_set:
                if is_multi:
                    for key, d in datadict.items():
                        subgraph.add_edge(u, v, key=key, **d)
                        ec += 1
                else:
                    subgraph.add_edge(u, v, **datadict)
                    ec += 1
    full_nodes = road_graph.number_of_nodes()
    full_edges = sp_cache.get("_full_edge_count")
    if full_edges is None:
        full_edges = road_graph.number_of_edges()
        sp_cache["_full_edge_count"] = full_edges
    fn, fe = max(1, full_nodes), max(1, full_edges)
    nc = len(valid_nodes)
    return subgraph, {
        "node_count": nc,
        "edge_count": ec,
        "corridor_node_count": nc,
        "corridor_edge_count": ec,
        "full_node_count": full_nodes,
        "full_edge_count": full_edges,
        "node_retention_ratio": nc / fn,
        "edge_retention_ratio": ec / fe,
        "pair_scope": pair_scope,
    }


def evaluate_cluster_invocation_gate(
    query,
    *,
    road_graph,
    camera_to_node,
    shortest_paths,
    rule_config=None,
    max_speed_mps=25.0,
    min_dwell_sec=30.0,
    max_span_m=1200.0,
    detour_ratio_threshold=1.05,
    detour_excess_m=0.0,
    candidate_bindings=None,
):
    """Evaluate C1 Level-1 cluster invocation gate (gamma_inv in Section III-A1).

    Checks whether the raw camera observation sequence exhibits any of the three conditions:
    (a) Inter-observation topological/kinematic conflict:
        - chronological conflict (dt < 0),
        - missing directed shortest-path connectivity on the road graph,
        - implied road-network speed exceeding max_speed_mps (v_imp > v_max), or
        - long-span observation gap d_SP(v_i, v_{i+1}) > max_span_m (default 1200.0 m);
    (b) Three-point mid-block detour anomaly:
        - d_SP(v_{i-1}, v_i) + d_SP(v_i, v_{i+1}) > detour_ratio_threshold * d_SP(v_{i-1}, v_{i+1})
          with excess path length exceeding detour_excess_m (default 0.0 m; 15.0 m on short-block grids);
    (c) Multi-candidate binding ambiguity or prolonged same-camera dwell:
        - |V(c_i)| > 1 in candidate_bindings, or c_i == c_{i+1} with dt > min_dwell_sec.
    """
    from utils.naive_recovery import _validated_path, normalize_naive_rules, recover_rule_shortest_path

    if isinstance(min_dwell_sec, bool) or not isinstance(min_dwell_sec, numbers.Real) or not math.isfinite(float(min_dwell_sec)) or float(min_dwell_sec) < 0:
        raise ValueError("min_dwell_sec must be a finite non-negative number")
    dwell_threshold = float(min_dwell_sec)
    span_threshold = None if max_span_m is None else float(max_span_m)
    detour_threshold = None if detour_ratio_threshold is None else float(detour_ratio_threshold)
    excess_threshold = max(0.0, float(detour_excess_m or 0.0))
    merged_rule_cfg = dict(rule_config or {})
    if max_speed_mps is not None:
        merged_rule_cfg["max_speed_mps"] = float(max_speed_mps)
    rules = normalize_naive_rules(merged_rule_cfg)
    query_dict = {"camera_trace": list(query)} if isinstance(query, (list, tuple)) else query
    rule_report = recover_rule_shortest_path(
        query_dict,
        road_graph=road_graph,
        camera_to_node=camera_to_node,
        shortest_paths=shortest_paths,
        rule_config=rules,
    )
    conflict_segments = []
    segments = rule_report["segments"]
    for seg in segments:
        reason = seg.get("reason")
        dt = seg.get("time_delta_sec")
        sp_len = seg.get("shortest_path_length_m")
        u, v = seg.get("from_node"), seg.get("to_node")
        if u == v:
            if dt is not None and dt > dwell_threshold:
                conflict_segments.append({
                    "segment_index": seg["segment_index"],
                    "trigger": "prolonged_same_camera_dwell",
                    "time_delta_sec": dt,
                })
            elif reason in {"reverse_time", "nonfinite_time_delta"}:
                conflict_segments.append({
                    "segment_index": seg["segment_index"],
                    "trigger": reason,
                    "time_delta_sec": dt,
                })
        else:
            if reason in {
                "reverse_time",
                "zero_time_delta",
                "nonfinite_time_delta",
                "missing_shortest_path",
                "invalid_cached_path",
                "missing_directed_edge",
                "excessive_speed",
                "nonfinite_implied_speed",
            }:
                conflict_segments.append({
                    "segment_index": seg["segment_index"],
                    "trigger": reason,
                    "time_delta_sec": dt,
                    "implied_speed_mps": seg.get("implied_speed_mps"),
                })
            elif span_threshold is not None and sp_len is not None and sp_len > span_threshold:
                conflict_segments.append({
                    "segment_index": seg["segment_index"],
                    "trigger": "long_span_gap",
                    "shortest_path_length_m": sp_len,
                    "time_delta_sec": dt,
                })

    if detour_threshold is not None and len(segments) >= 2:
        for idx in range(len(segments) - 1):
            seg_a, seg_b = segments[idx], segments[idx + 1]
            u0, u1, u2 = seg_a.get("from_node"), seg_a.get("to_node"), seg_b.get("to_node")
            d01 = seg_a.get("shortest_path_length_m")
            d12 = seg_b.get("shortest_path_length_m")
            if u0 is None or u1 is None or u2 is None or len({u0, u1, u2}) < 3 or d01 is None or d12 is None:
                continue
            entry_02 = shortest_paths.get((u0, u2))
            if entry_02 is None:
                continue
            rebuilt_02, err_02 = _validated_path(road_graph, entry_02, u0, u2)
            d02 = rebuilt_02[2] if err_02 is None and rebuilt_02 is not None else None
            if (
                d02 is not None
                and d02 > 0.0
                and (d01 + d12) > detour_threshold * d02
                and (d01 + d12 - d02) > excess_threshold
            ):
                conflict_segments.append({
                    "segment_index": idx,
                    "trigger": "three_point_detour_anomaly",
                    "detour_ratio": (d01 + d12) / d02,
                    "detour_excess_m": (d01 + d12) - d02,
                    "nodes": (u0, u1, u2),
                })

    bindings = candidate_bindings
    if bindings is None and isinstance(query, dict) and "camera_mapping_candidates" in query:
        bindings = {
            row["camera_id"]: [c["node"] for c in row.get("candidates", []) if isinstance(c, dict) and "node" in c]
            for row in query.get("camera_mapping_candidates", [])
            if isinstance(row, dict) and "camera_id" in row
        }
    if isinstance(bindings, dict):
        for obs_idx, (cam_id, _) in enumerate(query_dict.get("camera_trace", [])):
            cands = bindings.get(cam_id, [])
            if isinstance(cands, (list, tuple, set)) and len(set(cands)) > 1:
                conflict_segments.append({
                    "segment_index": obs_idx,
                    "trigger": "multi_candidate_binding_ambiguity",
                    "camera_id": cam_id,
                    "candidate_count": len(set(cands)),
                })

    return {
        "invoke_llm": bool(conflict_segments),
        "conflict_count": len(conflict_segments),
        "conflict_segments": conflict_segments,
        "segments": segments,
        "rule_config": rules,
        "min_dwell_sec": dwell_threshold,
        "max_span_m": span_threshold,
        "detour_ratio_threshold": detour_threshold,
        "detour_excess_m": excess_threshold,
    }


def execute_three_tier_corridor_verification(
    observations,
    interval_intents=None,
    *,
    road_graph,
    shortest_paths=None,
    max_speed_mps=25.0,
    max_span_m=1200.0,
    min_dwell_sec=30.0,
    detour_ratio_threshold=1.05,
    detour_excess_m=0.0,
):
    """Execute C2 three-tier deterministic corridor verification and timestamp interpolation (Section III-B2, Eq. 5).

    Args:
        observations: List of (road_node_id, timestamp_sec) tuples representing anchor observations.
        interval_intents: Optional dict mapping interval index i to a list of up to 2 via-node IDs [w_{i,1}, w_{i,2}].
        road_graph: Directed road network graph with edge 'length' attributes.
        shortest_paths: Optional precomputed shortest-path dict {(u, v): (nodes, cum_lengths, total_length)}.
        max_speed_mps: Kinematic speed threshold v_max (default 25.0 m/s).
        max_span_m: Optional maximum corridor span d_max in metres (default 1200.0 m).
        min_dwell_sec: Dwell collapse threshold Delta t_dwell (default 30.0 s).
        detour_ratio_threshold: Optional three-point detour ratio threshold eta_detour (default 1.05).
        detour_excess_m: Minimum excess path length Delta_detour in metres required to trigger a detour spike (default 0.0 m).
    """
    import networkx as nx

    if not observations:
        return {"trajectory": [], "segment_audits": [], "tier_counts": {1: 0, 2: 0, 3: 0}, "unresolved_count": 0}
    intents = interval_intents or {}
    sp_dict = shortest_paths if isinstance(shortest_paths, dict) else {}
    span_cap = None if max_span_m is None else float(max_span_m)
    detour_cap = None if detour_ratio_threshold is None else float(detour_ratio_threshold)
    excess_cap = max(0.0, float(detour_excess_m or 0.0))

    def _span_ok(dist_m):
        return span_cap is None or float(dist_m) <= span_cap

    def _edge_len(u, v):
        data = road_graph.get_edge_data(u, v)
        if data is None:
            return None
        edges = list(data.values()) if road_graph.is_multigraph() else [data]
        lengths = [float(d["length"]) for d in edges if isinstance(d, dict) and "length" in d and float(d["length"]) > 0]
        return min(lengths) if lengths else None

    def _shortest_path(u, v):
        if u == v:
            return [u], [0.0], 0.0
        cached = sp_dict.get((u, v))
        if isinstance(cached, (list, tuple)) and len(cached) == 3 and cached[0]:
            return list(cached[0]), list(cached[1]), float(cached[2])
        try:
            nodes = nx.shortest_path(road_graph, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        cum = [0.0]
        for a, b in zip(nodes, nodes[1:]):
            w = _edge_len(a, b)
            if w is None:
                return None
            cum.append(cum[-1] + w)
        return nodes, cum, cum[-1]

    def _interpolate(nodes, cum_lengths, t_start, t_end):
        total = cum_lengths[-1]
        if total <= 0.0 or len(nodes) <= 1:
            if float(t_end) > float(t_start):
                return [(nodes[0], float(t_start)), (nodes[-1], float(t_end))]
            return [(nodes[0], float(t_start))]
        dt = float(t_end) - float(t_start)
        return [(node, float(t_start) + dt * (dist / total)) for node, dist in zip(nodes, cum_lengths)]

    # Pre-filter chronological inversions and collapse short same-node dwell intervals
    cleaned = []
    for node, ts in observations:
        ts = float(ts)
        if not cleaned:
            cleaned.append((int(node), ts))
            continue
        prev_node, prev_ts = cleaned[-1]
        if ts < prev_ts:
            continue
        if int(node) == prev_node and (ts - prev_ts) <= float(min_dwell_sec):
            continue
        cleaned.append((int(node), ts))

    if len(cleaned) == 1:
        return {"trajectory": list(cleaned), "segment_audits": [], "tier_counts": {1: 0, 2: 0, 3: 0}, "unresolved_count": 0}

    trajectory = [cleaned[0]]
    audits = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    unresolved = 0
    idx = 0
    while idx < len(cleaned) - 1:
        u, t_start = cleaned[idx]
        v, t_end = cleaned[idx + 1]
        dt = t_end - t_start
        via = [int(w) for w in (intents.get(idx) or []) if w is not None][:2]
        accepted = None
        selected_tier = None

        # Tier 1: Constrained corridor path through via_nodes
        if via and dt > 0:
            chain = [u, *via, v]
            full_nodes = [u]
            full_cum = [0.0]
            ok = True
            for a, b in zip(chain, chain[1:]):
                sp = _shortest_path(a, b)
                if sp is None:
                    ok = False
                    break
                sub_nodes, sub_cum, _ = sp
                offset = full_cum[-1]
                for sn, sc in zip(sub_nodes[1:], sub_cum[1:]):
                    full_nodes.append(sn)
                    full_cum.append(offset + sc)
            if (
                ok
                and len(set(full_nodes)) == len(full_nodes)
                and _span_ok(full_cum[-1])
                and (full_cum[-1] / dt) <= float(max_speed_mps)
            ):
                accepted = (full_nodes, full_cum, t_start, t_end, 1)
                selected_tier = 1

        # Tier 2: Direct shortest-path fallback (unless flagged as a three-point detour spike)
        if accepted is None and dt > 0:
            sp = _shortest_path(u, v)
            is_detour_spike = False
            if (
                detour_cap is not None
                and sp is not None
                and (idx + 2) < len(cleaned)
                and len({u, v, cleaned[idx + 2][0]}) == 3
            ):
                v2, t_end2 = cleaned[idx + 2]
                dt2 = t_end2 - t_start
                sp_next = _shortest_path(v, v2)
                sp_bypass = _shortest_path(u, v2)
                if (
                    dt2 > 0
                    and sp_next is not None
                    and sp_bypass is not None
                    and sp_bypass[2] > 0.0
                    and _span_ok(sp_bypass[2])
                    and (sp_bypass[2] / dt2) <= float(max_speed_mps)
                    and (sp[2] + sp_next[2]) > detour_cap * sp_bypass[2]
                    and (sp[2] + sp_next[2] - sp_bypass[2]) > excess_cap
                ):
                    is_detour_spike = True
            if not is_detour_spike and sp is not None and _span_ok(sp[2]) and (sp[2] / dt) <= float(max_speed_mps):
                accepted = (sp[0], sp[1], t_start, t_end, 1)
                selected_tier = 2

        # Tier 3: Two-hop spike/detour bypass
        if accepted is None and (idx + 2) < len(cleaned):
            v2, t_end2 = cleaned[idx + 2]
            dt2 = t_end2 - t_start
            if dt2 > 0:
                sp2 = _shortest_path(u, v2)
                if sp2 is not None and _span_ok(sp2[2]) and (sp2[2] / dt2) <= float(max_speed_mps):
                    accepted = (sp2[0], sp2[1], t_start, t_end2, 2)
                    selected_tier = 3

        if accepted is not None:
            nodes, cum, ts0, ts1, hop_step = accepted
            pts = _interpolate(nodes, cum, ts0, ts1)
            trajectory.extend(pts[1:])
            tier_counts[selected_tier] += 1
            audits.append({
                "segment_index": idx,
                "from_node": u,
                "to_node": nodes[-1],
                "tier": selected_tier,
                "route_nodes": nodes,
                "path_length_m": cum[-1],
                "bypassed_observation_index": (idx + 1) if hop_step == 2 else None,
            })
            idx += hop_step
        else:
            unresolved += 1
            trajectory.append((v, t_end))
            audits.append({
                "segment_index": idx,
                "from_node": u,
                "to_node": v,
                "tier": None,
                "route_nodes": [u, v],
                "path_length_m": None,
                "bypassed_observation_index": None,
            })
            idx += 1

    return {
        "trajectory": trajectory,
        "segment_audits": audits,
        "tier_counts": tier_counts,
        "unresolved_count": unresolved,
    }


def extract_kinematic_dp_corridors(
    camera_trace,
    *,
    camera_to_node,
    shortest_paths,
    road_graph=None,
    max_intersection_cam_id=None,
    local_dist_m=380.0,
    stitch_dist_m=900.0,
    dwell_merge_sec=45.0,
    min_raw_support=4,
):
    """Separate kinematically consistent single-vehicle sub-corridors via two-stage DP (Section III-B1).

    Extracts smooth local chains (d_SP <= local_dist_m), stitches compatible non-overlapping
    subchains (d_SP <= stitch_dist_m), and collapses consecutive same-node dwell detections
    so co-clustered lookalike vehicles do not cross-jump across disjoint streets.
    """
    import networkx as nx

    def _sp_dist(u, v):
        if u == v:
            return 0.0
        if isinstance(shortest_paths, dict):
            entry = shortest_paths.get((u, v))
            if isinstance(entry, (list, tuple)) and len(entry) >= 3 and entry[0]:
                return float(entry[2])
            if isinstance(entry, (int, float)):
                return float(entry)
            sub = shortest_paths.get(u)
            if isinstance(sub, dict) and v in sub:
                return float(sub[v])
        if road_graph is not None:
            try:
                return float(nx.shortest_path_length(road_graph, u, v, weight="length"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return 1e9
        return 1e9

    pts = []
    for idx, (cam_raw, tm_raw) in enumerate(camera_trace or []):
        cam = int(cam_raw)
        if max_intersection_cam_id is not None and cam >= int(max_intersection_cam_id):
            continue
        if cam not in camera_to_node:
            continue
        pts.append((idx, cam, float(tm_raw), int(camera_to_node[cam])))
    pts.sort(key=lambda x: x[2])
    if not pts:
        return []

    remaining = list(range(len(pts)))
    subchains = []
    min_chain_len = max(2, min(4, int(min_raw_support)))
    while len(remaining) >= min_chain_len:
        n = len(remaining)
        dp = [1.0] * n
        prev = [-1] * n
        for j in range(n):
            _, _, tj, rj = pts[remaining[j]]
            for i in range(j):
                _, _, ti, ri = pts[remaining[i]]
                dt = tj - ti
                if dt < 0.0 or dt > 180.0:
                    continue
                d = _sp_dist(ri, rj)
                if d <= float(local_dist_m) and d <= 32.0 * dt + 85.0:
                    penalty = min(0.85, (d / max(float(local_dist_m), 1.0)) + (dt / 200.0))
                    score = dp[i] + 1.0 - penalty
                    if score > dp[j]:
                        dp[j] = score
                        prev[j] = i
        best_end = int(max(range(n), key=lambda k: dp[k]))
        path_idx = []
        cur = best_end
        while cur != -1:
            path_idx.append(remaining[cur])
            cur = prev[cur]
        path_idx.reverse()
        if len(path_idx) < min_chain_len:
            break
        subchains.append([pts[k] for k in path_idx])
        used = set(path_idx)
        remaining = [k for k in remaining if k not in used]

    subchains.sort(key=lambda ch: ch[0][2])
    merged = True
    while merged:
        merged = False
        best_pair = None
        best_cost = 1e18
        for i in range(len(subchains)):
            for j in range(len(subchains)):
                if i == j:
                    continue
                _, _, t_end, r_end = subchains[i][-1]
                _, _, t_start, r_start = subchains[j][0]
                dt = t_start - t_end
                if dt < 0.0 or dt > 200.0:
                    continue
                d = _sp_dist(r_end, r_start)
                if d <= float(stitch_dist_m) and d <= 18.0 * dt + 95.0:
                    cost = abs(d - 8.0 * dt) + 0.5 * d
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (i, j)
        if best_pair is not None:
            i, j = best_pair
            new_ch = subchains[i] + subchains[j]
            subchains = [ch for k, ch in enumerate(subchains) if k not in (i, j)] + [new_ch]
            subchains.sort(key=lambda ch: ch[0][2])
            merged = True

    valid_subchains = [ch for ch in subchains if len(ch) >= int(min_raw_support)]
    if not valid_subchains and subchains:
        valid_subchains = [max(subchains, key=len)]
    valid_subchains.sort(key=len, reverse=True)

    corridors = []
    for cid_idx, raw_ch in enumerate(valid_subchains):
        anchors = []
        cur_cam, cur_node = raw_ch[0][1], raw_ch[0][3]
        cur_tms = [raw_ch[0][2]]
        cur_idxs = [raw_ch[0][0]]
        for ridx, rcam, tm, rnode in raw_ch[1:]:
            if rnode == cur_node and (tm - cur_tms[-1]) <= float(dwell_merge_sec):
                cur_tms.append(tm)
                cur_idxs.append(ridx)
            else:
                anchors.append({
                    "camera_id": int(cur_cam),
                    "node": int(cur_node),
                    "time": float(sum(cur_tms) / len(cur_tms)),
                    "raw_count": len(cur_idxs),
                    "raw_indices": list(cur_idxs),
                })
                cur_cam, cur_node = rcam, rnode
                cur_tms = [tm]
                cur_idxs = [ridx]
        anchors.append({
            "camera_id": int(cur_cam),
            "node": int(cur_node),
            "time": float(sum(cur_tms) / len(cur_tms)),
            "raw_count": len(cur_idxs),
            "raw_indices": list(cur_idxs),
        })
        corridors.append({
            "corridor_id": cid_idx,
            "raw_support": len(raw_ch),
            "camera_trace": [(a["camera_id"], a["time"]) for a in anchors],
            "rid_trace": [(a["node"], a["time"]) for a in anchors],
            "anchors": anchors,
        })
    return corridors
