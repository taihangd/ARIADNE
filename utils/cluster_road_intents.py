"""C2 cluster-level trajectory recovery and structured road-intent (single_cluster / road_intents_v1) module.

Core architecture:
- The C2 mainline scheme is single_cluster + road_intents_v1:
  the LLM decides inter-observation connectivity support under a fixed observation order (via_nodes=[] selects the shortest path in the full directed road graph,
  via_nodes=null leaves the link as unknown when evidence is insufficient), with a single 8192-token generation call, zero repair retries, and zero response caching.
  Unparseable or unreachable intervals remain separate unknown fragments without heuristic stitching or fabricated timestamps.
"""
import copy
import json
import math
import re
import sys
import time
import networkx as nx


def _perf_counter():
    root = sys.modules.get("traj_rec_st_cot")
    if root is not None and hasattr(root, "time"):
        return root.time.perf_counter()
    return time.perf_counter()


def _strip_loose_markdown_fence(text):
    """Strip markdown code fences from LLM output, including unclosed trailing fences."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _cluster_track_road_graph(road_graph):
    """Copy only the declared road directions for shared use by public prompt evidence and deterministic rule execution."""
    from utils.road_time import edge_travel_directions

    if not isinstance(road_graph, nx.Graph) or not road_graph.number_of_nodes():
        return None, ["missing_road_graph"]
    directed = nx.MultiDiGraph()
    directed.add_nodes_from((node, dict(data)) for node, data in road_graph.nodes(data=True))
    issues = []
    for left, right, data in road_graph.edges(data=True):
        try:
            directions = edge_travel_directions(road_graph, left, right, data)
        except ValueError:
            issues.append("ambiguous_or_conflicting_edge_direction")
            continue
        for start, end in directions:
            directed.add_edge(start, end, **dict(data))
    return directed, sorted(set(issues))


def _cluster_road_intent_settings(config=None):
    """Freeze graph search budgets and provenance declarations; empirical speed limits are not treated as physical upper bounds."""
    from utils.naive_recovery import _finite

    settings = {"max_waypoints": 4, "max_search_expansions": 200000,
                "max_search_seconds": 2.0, "contract": {}}
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(settings):
            raise ValueError("Unsupported road intent configuration")
        settings.update(copy.deepcopy(config))
    if (type(settings["max_waypoints"]) is not int or not 0 <= settings["max_waypoints"] <= 4
            or type(settings["max_search_expansions"]) is not int
            or settings["max_search_expansions"] <= 0
            or not _finite(settings["max_search_seconds"]) or settings["max_search_seconds"] <= 0):
        raise ValueError("Road intent budgets must be finite, positive and bounded")
    contract = settings["contract"]
    allowed = {"spatial_scale_to_m", "time_scale_to_s", "units_reliable", "units_source",
               "direction_reliable", "direction_source", "coverage_reliable", "coverage_source",
               "geometry_reliable", "geometry_source", "vmax_reliable", "vmax_source", "vmax_mps",
               "turn_restrictions", "turn_reliable", "turn_source"}
    if not isinstance(contract, dict) or set(contract) - allowed:
        raise ValueError("Road intent contract accepts only public source declarations")
    for field in ("units", "direction", "coverage", "geometry", "vmax", "turn"):
        if field + "_reliable" in contract and type(contract[field + "_reliable"]) is not bool:
            raise ValueError("Source reliability must be an explicit boolean")
        contract.setdefault(field + "_reliable", False)
        source = contract.get(field + "_source")
        if contract[field + "_reliable"] and (not isinstance(source, str) or not source.strip()):
            contract[field + "_reliable"] = False
    for field in ("spatial_scale_to_m", "time_scale_to_s"):
        contract.setdefault(field, 1.0)
        if not _finite(contract[field]) or contract[field] <= 0:
            raise ValueError("Source scales must be finite and positive")
    # This protocol claims only conditional recovery within the supplied graph; even a reliable vmax does not replace full physical verification.
    contract.setdefault("vmax_mps", None)
    if contract["vmax_mps"] is not None and (not _finite(contract["vmax_mps"]) or contract["vmax_mps"] <= 0):
        raise ValueError("Declared vmax must be positive or unavailable")
    if contract["vmax_reliable"] and contract["vmax_mps"] is None:
        raise ValueError("Reliable vmax requires its source value")
    json.dumps(settings, allow_nan=False)
    return settings


def _cluster_hard_time_guard(contract):
    """Enable provenance-backed travel-time bounds only when all source reliability conditions hold under the no-turn-restriction model."""
    return (all(contract[field + "_reliable"] for field in
                ("units", "direction", "coverage", "geometry", "vmax", "turn"))
            and contract.get("turn_restrictions") == "none" and contract["vmax_mps"] is not None)


def _cluster_complete_road_context(road_graph, settings):
    """Expose all node IDs and directed arcs with bounded display precision while executing on exact raw coordinates and edge lengths."""
    from fractions import Fraction
    from utils.road_time import prepare_road_time_graph, _node_label, _node_value

    prepared = prepare_road_time_graph(road_graph)
    graph = prepared.get("graph")
    key = lambda node: json.dumps(_node_value(node), separators=(",", ":"))
    unit = 0.1
    origin = ([min(data[axis] for _, data in graph.nodes(data=True)) for axis in ("x", "y")]
              if graph is not None else [0.0, 0.0])
    nodes = [] if graph is None else [[_node_label(node),
                                      int(round((data["x"] - origin[0]) / unit)),
                                      int(round((data["y"] - origin[1]) / unit))]
                                     for node, data in sorted(graph.nodes(data=True), key=lambda item: key(item[0]))]
    length_unit = 0.001
    arcs = [] if graph is None else [[_node_label(left), _node_label(right),
                                     round(Fraction(float(data["weight"])) * 1000)]
                                    for left, right, data in sorted(graph.edges(data=True),
                                        key=lambda item: (key(item[0]), key(item[1])))]
    result = {"encoding": "complete_directed_rows_v2",
              "node_columns": ["node_id", "x_offset_steps", "y_offset_steps"],
              "coordinate_encoding": {"kind": "origin_relative_integer_xy_v1", "origin": origin,
                                      "unit_source_units": unit,
                                      "max_display_error_per_axis_source_units": unit / 2,
                                      "execution_coordinates": "original_graph"},
              "arc_columns": ["from_node", "to_node", "length_steps"],
              "length_encoding": {"kind": "integer_length_steps_v1", "unit_source_units": length_unit,
                                  "max_display_error_source_units": length_unit / 2,
                                  "execution_lengths": "original_graph",
                                  "display_lengths_are_not_hard_lower_bounds": True},
              "nodes": nodes, "arcs": arcs, "graph_hash": prepared.get("hash"),
              "node_count": len(nodes), "arc_count": len(arcs),
              "complete_supplied_graph": graph is not None, "truncated": False,
              "parallel_arcs": "minimum_declared_length_per_directed_endpoint_pair",
              "geometry": "all_nodes_with_quantized_display_xy; execution_uses_original_graph_geometry",
              "coverage_status": "source_declared" if settings["contract"]["coverage_reliable"] else "unknown",
              "direction_status": "source_declared" if settings["contract"]["direction_reliable"] else "unknown"}
    result["hard_time_guard_active"] = _cluster_hard_time_guard(settings["contract"])
    if graph is None:
        result["error"] = prepared.get("error", "unusable_road_graph_metadata")
    result["serialized_characters"] = len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return result


def build_cluster_track_context(query, *, road_graph, camera_to_node, shortest_paths,
                                planning_protocol="groups_v1", road_intent_config=None,
                                trajectory_scope="multi_track"):
    """Expand raw observation events from public fields only, without reading identity labels or evaluation scores."""
    from collections.abc import Mapping
    from utils.naive_recovery import _finite, _validated_path, _xy
    from utils.road_time import _node_label

    if planning_protocol not in {"groups_v1", "road_intents_v1"}:
        raise ValueError("Unsupported cluster planning protocol")
    if trajectory_scope not in {"single_cluster", "multi_track"}:
        raise ValueError("Unsupported cluster trajectory scope")
    if planning_protocol == "groups_v1" and road_intent_config is not None:
        raise ValueError("Road intent configuration requires road_intents_v1")
    allowed = {"camera_trace", "observed_points", "observation_provenance", "camera_positions"}
    if not isinstance(query, dict) or set(query) - allowed:
        raise ValueError("Cluster track query accepts only public observation fields")
    trace, points, provenance = (query.get(name) for name in
                                 ("camera_trace", "observed_points", "observation_provenance"))
    if (not isinstance(trace, list) or not trace or not isinstance(points, list)
            or len(points) != len(trace) or not isinstance(provenance, list)
            or len(provenance) != len(trace)):
        raise ValueError("Raw record expansion requires aligned observation provenance")
    positions = query.get("camera_positions", {})
    if not isinstance(positions, Mapping):
        raise ValueError("camera_positions must be an explicit mapping")
    graph, graph_issues = _cluster_track_road_graph(road_graph)
    mapping = camera_to_node if isinstance(camera_to_node, Mapping) else {}
    paths = shortest_paths if isinstance(shortest_paths, Mapping) else {}
    observations, seen = [], set()
    for index, (camera_point, point, member) in enumerate(zip(trace, points, provenance)):
        if (not isinstance(camera_point, (list, tuple)) or len(camera_point) != 2
                or not isinstance(point, (list, tuple)) or len(point) != 3
                or not all(_finite(value) for value in point)
                or not _finite(camera_point[1]) or camera_point[1] != point[2]
                or not isinstance(member, dict)):
            raise ValueError("Camera observations and provenance must align")
        camera, prepared_time = camera_point
        aggregation = member.get("aggregation")
        records, times = member.get("raw_record_indices"), member.get("transformed_times")
        raw_times = member.get("raw_times_s")
        if (member.get("input_index") != index or member.get("camera_id") != camera
                or member.get("prepared_time") != prepared_time
                or not isinstance(aggregation, dict) or aggregation.get("exact_reproduction") is not True
                or not isinstance(records, list) or not records or not isinstance(times, list)
                or not isinstance(raw_times, list) or len(records) != len(times) or len(records) != len(raw_times)
                or any(type(record) is not int or record < 0 for record in records)
                or not all(_finite(stamp) for stamp in times + raw_times)
                or not math.isclose(sum(times) / len(times), prepared_time, rel_tol=0, abs_tol=1e-6)):
            raise ValueError("Raw record expansion requires exact reproduced provenance")
        conversion = aggregation.get("time_conversion")
        if conversion is not None:
            if (not isinstance(conversion, dict) or set(conversion) != {"fps", "offset_seconds"}
                    or not _finite(conversion["fps"]) or conversion["fps"] <= 0
                    or not _finite(conversion["offset_seconds"])
                    or any(not math.isclose(raw + conversion["offset_seconds"], transformed,
                                             rel_tol=0, abs_tol=1e-6)
                           for raw, transformed in zip(raw_times, times))):
                raise ValueError("Record time conversion must reproduce transformed timestamps")
            conversion = {key: float(conversion[key]) for key in ("fps", "offset_seconds")}
        camera_xy = positions.get(camera, positions.get(str(camera)))
        if camera_xy is not None and (not isinstance(camera_xy, (list, tuple)) or len(camera_xy) != 2
                                      or not all(_finite(value) for value in camera_xy)):
            raise ValueError("Camera coordinates must be finite planar coordinates")
        for record, raw_stamp, stamp in zip(records, raw_times, times):
            if record in seen:
                raise ValueError("Raw record indices must be unique within the cluster")
            seen.add(record)
            try:
                node = mapping[camera]
                node_xy = _xy(graph, node) if graph is not None else None
                node_label = _node_label(node)
            except (KeyError, TypeError, ValueError):
                node, node_xy, node_label = None, None, None
            observations.append({
                "obs_id": f"record:{record}", "record_index": record, "input_index": index,
                "camera_id": camera, "time": float(stamp), "raw_time_s": float(raw_stamp),
                "time_conversion": conversion,
                "camera_xy": [float(value) for value in camera_xy] if camera_xy is not None else None,
                "camera_position_status": "supplied" if camera_xy is not None else "unknown",
                "prepared_observation_xy": [float(value) for value in point[:2]],
                "measurement_semantics": "camera_location_not_vehicle_position",
                "road_node": node_label, "road_node_xy": list(node_xy) if node_xy else None,
                "mapping_status": "conditional" if node_xy else "unknown",
                "identity_status": "unknown", "time_reliability": "unknown",
                "time_semantics": "transformed_record_timestamp",
            })
    # Stable sorting preserves distinct simultaneous events without duplicating legacy aggregated records.
    observations.sort(key=lambda item: item["time"])
    evidence = []
    for index, first in enumerate(observations):
        for second in observations[index + 1:]:
            dt = second["time"] - first["time"]
            item = {"from_obs_id": first["obs_id"], "to_obs_id": second["obs_id"],
                    "time_delta_s": dt if _finite(dt) else None, "status": "unknown", "issues": [],
                    "shortest_path_length_m": None, "required_average_speed_mps": None,
                    "distance_source": None, "globally_shortest_route_status": "unknown",
                    "detour_ratio": None, "route_nodes": None,
                    "speed_assumption_mps": 65.0, "exceeds_speed_assumption": None}
            evidence.append(item)
            if not _finite(dt):
                item["issues"] = ["nonfinite_time_delta"]
                continue
            if first["mapping_status"] == "unknown" or second["mapping_status"] == "unknown":
                item["issues"] = ["unknown_observation_mapping_or_graph"]
                continue
            start, end = mapping[first["camera_id"]], mapping[second["camera_id"]]
            if start == end:
                rebuilt, error = ([start], [0.0], 0.0), None
            else:
                entry = paths.get((start, end))
                rebuilt, error = _validated_path(graph, entry, start, end) if entry is not None else (
                    None, "missing_shortest_path")
            if error:
                item["issues"] = [error]
                continue
            length = float(rebuilt[2])
            chord = math.dist(first["road_node_xy"], second["road_node_xy"])
            speed = length / dt if dt > 0 else (0.0 if length == 0 else None)
            ratio = length / chord if chord > 0 else None
            if speed is not None and not _finite(speed):
                speed = None
                item["issues"].append("nonfinite_required_average_speed")
            if ratio is not None and not _finite(ratio):
                ratio = None
                item["issues"].append("nonfinite_detour_ratio")
            item.update(status="conditional_cached_route", shortest_path_length_m=length,
                        distance_source="validated_cached_route_edge_lengths",
                        required_average_speed_mps=speed,
                        detour_ratio=ratio,
                        route_nodes=[_node_label(node) for node in rebuilt[0]],
                        exceeds_speed_assumption=speed > 65 if speed is not None else None)
            if dt == 0:
                item["issues"].append("equal_time_events_do_not_establish_a_connection")
    context = {"schema": "cluster_tracks_v1", "observations": observations,
            "trajectory_scope": trajectory_scope,
            "task_scope": {"trajectory_unit": "one_input_cluster" if trajectory_scope == "single_cluster"
                           else "event_partition_hypotheses",
                           "association_basis": "declared_cluster_membership" if trajectory_scope == "single_cluster"
                           else "model_hypotheses",
                           "physical_or_identity_guarantee": False},
            "event_order": "stable_transformed_time_then_original_member_order",
            "pair_evidence": evidence, "graph_issues": graph_issues,
            "assumptions": {"units": "declared planar metres and seconds",
                            "speed_assumption_mps": 65.0,
                            "speed_is_a_diagnostic_not_an_identity_filter": True,
                            "identity_and_real_world_physical_validity": "unknown",
                            "appearance_association_evidence": "unavailable"},
            "output_contract": {"tracks": [{"obs_ids": ["record:<assigned_index>"]}],
                                "unassigned_obs_ids": ["record:<unassigned_index>"]}}
    if planning_protocol == "road_intents_v1":
        settings = _cluster_road_intent_settings(road_intent_config)
        context.update(planning_protocol=planning_protocol, road_intent_settings=settings,
                       road_context=_cluster_complete_road_context(road_graph, settings))
        context["output_contract"]["tracks"][0]["obs_ids"] = [
            "record:<assigned_start_index>", "record:<assigned_end_index>"]
        context["output_contract"]["tracks"][0]["interval_intents"] = [{
            "from_obs_id": "record:<assigned_start_index>",
            "to_obs_id": "record:<assigned_end_index>", "via_nodes": []}]
        context["road_intent_semantics"] = (
            "Each adjacent event pair has exactly one intent. [] selects the complete-graph shortest "
            "route; null leaves the connection unknown. "
            + ("No waypoints are allowed; the LLM decides event grouping and connection support only. "
               if settings["max_waypoints"] == 0 else
               f"At most {settings['max_waypoints']} supplied node IDs constrain the route in order. ")
            + "Source times and execution graph lengths are immutable; displayed lengths are approximate.")
    if trajectory_scope == "single_cluster":
        ids = [obs["obs_id"] for obs in observations]
        context["output_contract"] = {"tracks": [{"obs_ids": ids}], "unassigned_obs_ids": []}
        if planning_protocol == "road_intents_v1":
            context["output_contract"]["tracks"][0]["interval_intents"] = [
                {"from_obs_id": left, "to_obs_id": right, "via_nodes": []}
                for left, right in zip(ids, ids[1:])]
            context["road_intent_semantics"] = context["road_intent_semantics"].replace(
                "the LLM decides event grouping and connection support only",
                "event membership is fixed and the LLM decides connection support only")
    return context


def build_cluster_track_prompt(context, mode):
    """Share identical cluster facts and grouping interfaces across arms while varying only reasoning guidance; no repair calls are issued."""
    if mode not in {"plain", "evidence"}:
        raise ValueError("Cluster track mode must be plain or evidence")
    single_cluster = context.get("trajectory_scope", "multi_track") == "single_cluster"
    guidance = (("Recover the road trajectory for the fixed observation chain." if single_cluster else
                 "Group the events into supported trajectory hypotheses.")
                if mode == "plain" else
                "Internally reason through the supplied route directions, lengths and event times for each "
                "proposed connection. Use them to assess support and uncertainty. "
                + ("Observation membership is already fixed by the input cluster. " if single_cluster else
                   "Do not use route feasibility to infer identity. ") +
                "Do not include that reasoning in the response.")
    road_intents = context.get("planning_protocol") == "road_intents_v1"
    zero_waypoints = road_intents and context["road_intent_settings"]["max_waypoints"] == 0
    route_decision = ("via_nodes must be [] for a supported connection or null for an unsupported "
                      "connection. " + ("Decide connection support only; " if single_cluster else
                                         "Decide event grouping and connection support only; ") + "a program "
                      "chooses the road path. pair_evidence.route_nodes are input evidence; never "
                      "copy them into via_nodes. Nonempty via_nodes arrays are forbidden. " if zero_waypoints else
                      f"Choose zero to {context.get('road_intent_settings', {}).get('max_waypoints', 4)} "
                      "publicly supplied road-node IDs as ordered via_nodes to express your route decision. ")
    output = ("Return only one complete JSON object with exactly tracks:[{obs_ids:[...],"
              "interval_intents:[{from_obs_id:...,to_obs_id:...,via_nodes:"
              + ("[]" if zero_waypoints else "[...]") + "}]}] and "
              "unassigned_obs_ids:[...]. An n-event track has exactly n-1 interval intents, "
              "one per adjacent pair in the same order; a singleton has an empty interval_intents list. "
              "Interval endpoint IDs reference the track's obs_ids; they do not assign events again. "
              "Keep all intervals of the same supported trajectory in one track, rather than "
              "separate tracks for its adjacent intervals. " + route_decision + "[] requests a shortest "
              "route in the complete directed graph; null leaves a connection unsupported. "
              "Do not output full paths, change source times or edge lengths, or invent nodes. "
              "All supplied branches are available; there is no fixed path candidate pool. "
              "A program expands your route intentions and allocates node times by length; "
              + ("that interpolation does not establish real driving times. " if single_cluster else
                 "that interpolation does not establish real driving times or vehicle identity. ")
              if road_intents else
              "Return only one complete JSON object with exactly tracks:[{obs_ids:[...]}] and "
              "unassigned_obs_ids:[...]. ")
    scope_instruction = (
        "Recover one road trajectory for this input cluster. Upstream clustering defines one trajectory "
        "hypothesis: exactly one nonempty track is required. Retain all observations by default, in the "
        "supplied order. You may exclude an observation from the recovered path only when the public "
        "road and temporal context supports treating it as an anomaly candidate; put its original obs_id "
        "in unassigned_obs_ids, so the source observation remains recorded. This is uncertain filtering, "
        "not an identity decision or proof of sensor noise. Unknown connection support alone is not a "
        "reason to exclude an observation. Do not split the cluster into multiple tracks, exclude all "
        "events, use evaluation answers, or change source timestamps. This is the task's cluster-membership "
        "assumption, not a physical guarantee. An unsupported connection stays unknown within this same "
        "track; retain both endpoint observations. Camera locations and road-node coordinates have "
        "different meanings. " if single_cluster else
        "Recover all distinguishable trajectory hypotheses in this supplied cluster. "
        "The cluster may contain several trajectories. No event has a certified identity; "
        "do not invent vehicle identities, change times, or force every event into one trajectory. "
        "Camera locations and road-node coordinates have different meanings. "
        "Use singleton tracks or unassigned_obs_ids when a connection is unsupported. "
        "A singleton is an unverified one-event hypothesis, not an identified vehicle. ")
    if single_cluster:
        output += ("Exactly one nonempty track is required; its obs_ids must be an ordered subsequence "
                   "of the supplied events. Every excluded event must appear in unassigned_obs_ids. "
                   "The full observation chain shown in output_contract is the default retained chain. ")
    return (scope_instruction + "Unknown support means insufficient evidence, not proven impossibility. "
            + output + "Across all tracks[*].obs_ids and unassigned_obs_ids, every supplied obs_id "
            "must occur exactly once. Each track must be "
            "nonempty and follow the supplied event order; retain distinct equal-time events. "
            "There will be zero correction calls.\n" + guidance + "\n[CLUSTER_TRACK_FACTS]\n" +
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n[FINAL_OUTPUT_CONTRACT]\n" + output
            + "Every supplied obs_id must occur exactly once; retain distinct equal-time events. "
            "Output only the JSON object. No explanations, Markdown fences, calculations, or alternative answers.")


def parse_cluster_track_groups(text, context, *, track_protocol=None):
    """Parse strict JSON event partitions without inventing, omitting, duplicating, or reordering events within any track."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object keys are forbidden")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constants are forbidden")

    if not isinstance(text, str):
        raise ValueError("Cluster track groups require a complete JSON string")
    # Accept only a complete outer JSON wrapper; never extract embedded JSON from prose, truncated outputs, or multi-object text.
    text = text.strip()
    if re.fullmatch(r"```(?:json)?\s*[\s\S]*?\s*```", text, flags=re.IGNORECASE):
        text = _strip_loose_markdown_fence(text)
    groups = json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if not isinstance(groups, dict) or set(groups) != {"tracks", "unassigned_obs_ids"}:
        raise ValueError("Cluster groups require exactly tracks and unassigned_obs_ids")
    if track_protocol not in {None, "direct_path_v1"}:
        raise ValueError("Unsupported cluster response protocol")
    direct_path = track_protocol == "direct_path_v1"
    observations = context["observations"]
    rank = {obs["obs_id"]: index for index, obs in enumerate(observations)}
    if len(rank) != len(observations):
        raise ValueError("Context event IDs must be unique")
    tracks, unassigned = groups["tracks"], groups["unassigned_obs_ids"]
    if (not isinstance(tracks, list) or len(tracks) > len(rank) or not isinstance(unassigned, list)):
        raise ValueError("Cluster groups must contain bounded lists")
    trajectory_scope = context.get("trajectory_scope", "multi_track")
    if trajectory_scope not in {"single_cluster", "multi_track"}:
        raise ValueError("Unsupported cluster trajectory scope")
    if trajectory_scope == "single_cluster":
        if len(tracks) != 1:
            raise ValueError("Single-cluster recovery requires exactly one nonempty track")
    covered = []
    road_intents = not direct_path and context.get("planning_protocol") == "road_intents_v1"
    public_nodes = ({json.dumps(row[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                     for row in context.get("road_context", {}).get("nodes", [])} if direct_path else set())
    for track in tracks:
        expected = ({"obs_ids", "node_time_path"} if direct_path else
                    {"obs_ids", "interval_intents"} if road_intents else {"obs_ids"})
        if not isinstance(track, dict) or set(track) != expected:
            raise ValueError("Each track requires exactly " + ", ".join(sorted(expected)))
        ids = track["obs_ids"]
        if (not isinstance(ids, list) or not ids or any(not isinstance(obs_id, str) for obs_id in ids)
                or any(obs_id not in rank for obs_id in ids)):
            raise ValueError("Each track requires known nonempty event IDs")
        if any(rank[left] >= rank[right] for left, right in zip(ids, ids[1:])):
            raise ValueError("Track events must retain the supplied chronological order")
        if direct_path:
            from utils.naive_recovery import _finite
            path = track["node_time_path"]
            if not isinstance(path, list) or not path:
                raise ValueError("Direct tracks require a nonempty node_time_path")
            for point in path:
                if (not isinstance(point, list) or len(point) != 2
                        or type(point[0]) not in (int, float, str, list) or not _finite(point[1])):
                    raise ValueError("Direct paths require public node IDs and finite numeric times")
                try:
                    node_key = json.dumps(point[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                except (ValueError, OverflowError) as exc:
                    raise ValueError("Direct paths require finite public node IDs") from exc
                if node_key not in public_nodes:
                    raise ValueError("Direct path node ID was not publicly supplied")
        if road_intents:
            intents = track["interval_intents"]
            if not isinstance(intents, list) or len(intents) != len(ids) - 1:
                raise ValueError("Every adjacent event pair requires one road intent")
            for (left, right), intent in zip(zip(ids, ids[1:]), intents):
                if (not isinstance(intent, dict)
                        or set(intent) != {"from_obs_id", "to_obs_id", "via_nodes"}
                        or intent["from_obs_id"] != left or intent["to_obs_id"] != right):
                    raise ValueError("Road intents must reference adjacent events in their supplied order")
                via = intent["via_nodes"]
                if via is not None and (not isinstance(via, list)
                        or len(via) > context["road_intent_settings"]["max_waypoints"]
                        or any(type(node) not in (int, float, str, list) for node in via)):
                    raise ValueError("Road intents require bounded node IDs or null")
        covered.extend(ids)
    if any(not isinstance(obs_id, str) or obs_id not in rank for obs_id in unassigned):
        raise ValueError("Unassigned event IDs must be known")
    covered.extend(unassigned)
    if len(covered) != len(rank) or len(set(covered)) != len(covered) or set(covered) != set(rank):
        raise ValueError("Every event must appear exactly once")
    return groups


def _cluster_bounded_shortest_path(graph, start, end, budget, *, max_distance=None,
                                  remaining_lower_bound=0.0, geometric_lower_bound=False):
    """Run deterministic Dijkstra search over non-negative edge lengths on the full graph under a shared per-cluster search budget."""
    import heapq
    from itertools import count
    from utils.road_time import _node_value

    if start == end:
        return ([start], [0.0], 0.0), None
    order = lambda node: json.dumps(_node_value(node), separators=(",", ":"))
    serial = count()
    queue, distances, previous = [(0.0, next(serial), start)], {start: 0.0}, {}
    pruned = False
    while queue:
        if _perf_counter() >= budget["deadline"]:
            budget["truncated"] = True
            return None, "search_time_budget_exhausted"
        distance, _, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if budget["expansions"] >= budget["max_expansions"]:
            budget["truncated"] = True
            return None, "search_expansion_budget_exhausted"
        budget["expansions"] += 1
        if node == end:
            nodes = [end]
            while nodes[-1] != start:
                nodes.append(previous[nodes[-1]])
            nodes.reverse()
            lengths = [0.0]
            for left, right in zip(nodes, nodes[1:]):
                lengths.append(lengths[-1] + graph[left][right]["weight"])
            return (nodes, lengths, lengths[-1]), None
        for target in sorted(graph.successors(node), key=order):
            candidate = distance + graph[node][target]["weight"]
            if not math.isfinite(candidate):
                return None, "nonfinite_path_length"
            if max_distance is not None:
                goal_bound = (math.hypot(graph.nodes[target]["x"] - graph.nodes[end]["x"],
                                         graph.nodes[target]["y"] - graph.nodes[end]["y"])
                              if geometric_lower_bound else 0.0)
                if candidate + goal_bound + remaining_lower_bound > max_distance + 1e-9:
                    pruned = True
                    budget["time_bound_pruned_edges"] += 1
                    continue
            if candidate < distances.get(target, math.inf):
                distances[target], previous[target] = candidate, node
                heapq.heappush(queue, (candidate, next(serial), target))
    return None, "fixed_anchor_time_budget_violation" if pruned else "no_route_in_supplied_directed_graph"


def _cluster_road_intent_run(run_ids, group, events, mapping, graph, settings, budget, public_nodes):
    """Validate frozen waypoint intents and expand them on the full graph; split into separate fragments on failure without falling back to alternative intents."""
    from utils.road_time import _node_label

    bounded_shortest_path = getattr(
        sys.modules.get("traj_rec_st_cot"), "_cluster_bounded_shortest_path", _cluster_bounded_shortest_path
    )
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    intents = {(item["from_obs_id"], item["to_obs_id"]): item for item in group["interval_intents"]}
    trajectory = [(mapping[events[run_ids[0]]["camera_id"]], events[run_ids[0]]["time"])]
    segments = []
    contract = settings["contract"]
    hard_guard = _cluster_hard_time_guard(contract)
    # Use Euclidean distance as an A* heuristic lower bound only when it never exceeds any supplied arc length; otherwise use a zero heuristic.
    chord_bounds = hard_guard and all(data["weight"] + 1e-9 >= math.hypot(
        graph.nodes[u]["x"] - graph.nodes[v]["x"], graph.nodes[u]["y"] - graph.nodes[v]["y"])
        for u, v, data in graph.edges(data=True))
    for index, (left, right) in enumerate(zip(run_ids, run_ids[1:])):
        first, second = events[left], events[right]
        start, end = mapping[first["camera_id"]], mapping[second["camera_id"]]
        dt, via = second["time"] - first["time"], intents[left, right]["via_nodes"]
        segment = {"segment_index": index, "from_obs_id": left, "to_obs_id": right,
                   "from_node": _node_label(start), "to_node": _node_label(end), "via_nodes": copy.deepcopy(via),
                   "resolved": False, "status": "unknown", "interpolated_count": 0,
                   "time_delta_source_units": dt if math.isfinite(dt) else None,
                   "full_graph_shortest_length_m": None, "chosen_route_length_m": None,
                   "source_declared_minimum_time_s": None,
                   "hard_time_guard_active": hard_guard,
                   "required_average_speed_mps": None, "speed_assumption_mps": 65.0,
                   "exceeds_speed_assumption": None, "physical_status": "unknown",
                   "time_status": "unknown", "graph_coverage_status": "source_declared"
                        if contract["coverage_reliable"] else "unknown",
                   "direction_status": "source_declared" if contract["direction_reliable"] else "unknown",
                   "units_status": "source_declared" if contract["units_reliable"] else "unknown",
                   "turn_status": "source_declared" if contract["turn_reliable"] else "unknown",
                   "search_truncated": False, "search_expansions": 0}
        segments.append(segment)
        previous_expansions = budget["expansions"]
        if not math.isfinite(dt) or dt <= 0:
            segment["reason"] = "nonfinite_time_delta" if not math.isfinite(dt) else "zero_time_delta" if dt == 0 else "reverse_time"
        elif via is None:
            segment["reason"] = "model_unknown_route_intent"
        elif any(encode(node) not in public_nodes for node in via):
            segment["reason"] = "waypoint_not_in_public_graph"
        else:
            shortest, error = bounded_shortest_path(graph, start, end, budget)
            if error:
                segment["reason"] = error
            else:
                minimum_m = shortest[2] * contract["spatial_scale_to_m"]
                segment["full_graph_shortest_length_m"] = minimum_m if math.isfinite(minimum_m) else None
                if hard_guard and math.isfinite(minimum_m):
                    minimum_time = minimum_m / contract["vmax_mps"]
                    segment["source_declared_minimum_time_s"] = minimum_time if math.isfinite(minimum_time) else None
                route = shortest
                if via or hard_guard:
                    anchors = [start, *(public_nodes[encode(node)] for node in via), end]
                    minimum_legs = []
                    if hard_guard:
                        for origin, target in zip(anchors, anchors[1:]):
                            leg, error = ((shortest, None) if not via else
                                bounded_shortest_path(graph, origin, target, budget))
                            if error:
                                break
                            minimum_legs.append(leg[2])
                    nodes, lengths, distance = [start], [0.0], 0.0
                    allowed_distance = (dt * contract["time_scale_to_s"] * contract["vmax_mps"]
                                        / contract["spatial_scale_to_m"] if hard_guard else None)
                    if hard_guard and not math.isfinite(allowed_distance):
                        error = "invalid_scaled_time_budget"
                    for leg_index, (origin, target) in enumerate(zip(anchors, anchors[1:])):
                        if error:
                            break
                        leg, error = bounded_shortest_path(graph, origin, target, budget,
                            max_distance=allowed_distance - distance if hard_guard else None,
                            remaining_lower_bound=sum(minimum_legs[leg_index + 1:]) if hard_guard else 0.0,
                            geometric_lower_bound=chord_bounds)
                        if error:
                            break
                        nodes.extend(leg[0][1:])
                        lengths.extend(distance + length for length in leg[1][1:])
                        distance += leg[2]
                    route = (nodes, lengths, distance) if error is None else None
                if error:
                    segment["reason"] = error
                else:
                    nodes, lengths, distance = route
                    if (not math.isfinite(distance) or distance + 1e-6 < shortest[2]
                            or not math.isfinite(distance * contract["spatial_scale_to_m"])):
                        segment["reason"] = "invalid_route_distance"
                    else:
                        length_m = distance * contract["spatial_scale_to_m"]
                        seconds = dt * contract["time_scale_to_s"]
                        speed = length_m / seconds if math.isfinite(seconds) and seconds > 0 else math.inf
                        segment.update(chosen_route_length_m=length_m,
                            required_average_speed_mps=speed if math.isfinite(speed) else None,
                            exceeds_speed_assumption=speed > 65 if math.isfinite(speed) else None)
                        if not math.isfinite(seconds) or seconds <= 0:
                            segment["reason"] = "invalid_scaled_time_delta"
                        elif hard_guard and (not math.isfinite(speed) or speed > contract["vmax_mps"] + 1e-9):
                            segment["reason"] = "fixed_anchor_time_budget_violation"
                        else:
                            # Preserve all nodes along zero-length coincident edges while keeping exact original anchor timestamps.
                            internal = [(node, first["time"] + dt * (length / distance if distance else 0.0))
                                        for node, length in zip(nodes[1:-1], lengths[1:-1])]
                            trajectory.extend(internal)
                            segment.update(resolved=True, status="conditional", reason="route_intent_expanded",
                                interpolated_count=len(internal), time_status="conditional_length_interpolation",
                                route_nodes=[_node_label(node) for node in nodes])
        segment["search_expansions"] = budget["expansions"] - previous_expansions
        segment["search_truncated"] = segment.get("reason", "").startswith("search_")
        trajectory.append((end, second["time"]))
    return {"trajectory": trajectory, "segments": segments}


def execute_cluster_track_groups(groups, context, *, road_graph, camera_to_node, shortest_paths,
                                 rule_config=None):
    """Execute frozen rule-based or waypoint-intent expansion while keeping unresolved links as separate fragments."""
    from collections.abc import Mapping
    from utils.naive_recovery import normalize_naive_rules, recover_rule_shortest_path, _xy
    from utils.road_time import _node_label

    groups = parse_cluster_track_groups(json.dumps(groups, allow_nan=False), context)
    trajectory_scope = context.get("trajectory_scope", "multi_track")
    association_status = "cluster_membership_assumption" if trajectory_scope == "single_cluster" else "hypothesis"
    road_intents = context.get("planning_protocol") == "road_intents_v1"
    rules = normalize_naive_rules(rule_config)
    graph, graph_issues = _cluster_track_road_graph(road_graph)
    budget, settings, public_nodes = None, None, None
    if road_intents:
        from utils.road_time import prepare_road_time_graph
        settings = _cluster_road_intent_settings(context["road_intent_settings"])
        prepared = prepare_road_time_graph(road_graph)
        graph = prepared.get("graph")
        if prepared.get("hash") != context["road_context"].get("graph_hash"):
            graph, graph_issues = None, ["frozen_road_graph_changed"]
        elif graph is None:
            graph_issues = [prepared.get("error", "unusable_road_graph_metadata")]
        encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        public_nodes = {} if graph is None else {encode(_node_label(node)): node for node in graph}
        if graph is not None and set(public_nodes) != {encode(row[0]) for row in context["road_context"]["nodes"]}:
            graph, graph_issues = None, ["public_road_node_coverage_changed"]
        budget = {"deadline": _perf_counter() + settings["max_search_seconds"], "expansions": 0,
                  "max_expansions": settings["max_search_expansions"], "truncated": False,
                  "time_bound_pruned_edges": 0}
    mapping = camera_to_node if isinstance(camera_to_node, Mapping) else {}
    paths = shortest_paths if isinstance(shortest_paths, Mapping) else {}
    events = {obs["obs_id"]: obs for obs in context["observations"]}
    tracks, fragments, unresolved, unassigned, route_audits = [], [], [], [], []

    def unassign(obs, reason):
        unassigned.append({**copy.deepcopy(obs), "execution_status": "unknown", "issues": [reason],
                           **({"track_id": "track:0", "association_status": association_status}
                              if trajectory_scope == "single_cluster" else {})})

    for obs_id in groups["unassigned_obs_ids"]:
        unassign(events[obs_id], "model_excluded_uncertain" if trajectory_scope == "single_cluster" else "model_unassigned")

    def finish(track_id, obs_ids, trajectory):
        if not obs_ids:
            return
        path = [[*_xy(graph, node), float(stamp)] for node, stamp in trajectory]
        fragments.append({"fragment_id": f"fragment:{len(fragments)}", "track_id": track_id,
                          "obs_ids": list(obs_ids),
                          "observations": [copy.deepcopy(events[obs_id]) for obs_id in obs_ids],
                          "trajectory": [[_node_label(node), float(stamp)] for node, stamp in trajectory],
                          "path": path, "identity_status": "unknown", "physical_status": "unknown",
                          "execution_status": "completed", "timing_policy": "original_events_plus_length_interpolation"
                              if road_intents else "original_events_plus_rule_interpolation",
                          **({"time_status": "conditional_length_interpolation"} if road_intents else {})})

    for group_index, group in enumerate(groups["tracks"]):
        track_id = f"track:{group_index}"
        ids = group["obs_ids"]
        tracks.append({"track_id": track_id, "obs_ids": list(ids), "identity_status": "unknown",
                       "association_status": association_status})
        runs, run, unusable_ids = [], [], set()
        for obs_id in ids:
            obs = events[obs_id]
            try:
                node = mapping[obs["camera_id"]]
                usable = graph is not None and _xy(graph, node) is not None
                if road_intents:
                    usable = usable and encode(_node_label(node)) == encode(obs["road_node"])
            except (KeyError, TypeError):
                usable = False
            if not usable:
                unusable_ids.add(obs_id)
                if run:
                    runs.append(run)
                    run = []
                unassign(obs, "unknown_observation_mapping_or_graph")
            else:
                run.append(obs_id)
        if run:
            runs.append(run)
        for left, right in zip(ids, ids[1:]):
            if any(events[obs_id]["mapping_status"] == "unknown" or (road_intents and obs_id in unusable_ids)
                   for obs_id in (left, right)):
                unresolved.append({"track_id": track_id, "from_obs_id": left, "to_obs_id": right,
                                   "reason": "unknown_observation_mapping_or_graph", "status": "unknown"})
        for run_ids in runs:
            query = {"camera_trace": [[events[obs_id]["camera_id"], events[obs_id]["time"]]
                                      for obs_id in run_ids]}
            result = (_cluster_road_intent_run(run_ids, group, events, mapping, graph, settings, budget, public_nodes)
                      if road_intents else recover_rule_shortest_path(query, road_graph=graph, camera_to_node=mapping,
                                                                      shortest_paths=paths, rule_config=rules))
            if road_intents:
                route_audits.extend({"track_id": track_id, **segment} for segment in result["segments"])
            trajectory = result["trajectory"]
            current_ids, current_path, offset = [run_ids[0]], [trajectory[0]], 0
            for index, segment in enumerate(result["segments"]):
                count = segment["interpolated_count"]
                endpoint = trajectory[offset + count + 1]
                if segment["resolved"]:
                    current_ids.append(run_ids[index + 1])
                    current_path.extend(trajectory[offset + 1:offset + count + 2])
                else:
                    finish(track_id, current_ids, current_path)
                    current_ids, current_path = [run_ids[index + 1]], [endpoint]
                    unresolved.append({"track_id": track_id, "from_obs_id": run_ids[index],
                                       "to_obs_id": run_ids[index + 1], "reason": segment["reason"],
                                       "status": "unknown", "rule_audit": segment})
                offset += count + 1
            finish(track_id, current_ids, current_path)
    assigned = [obs_id for fragment in fragments for obs_id in fragment["obs_ids"]]
    assigned.extend(obs["obs_id"] for obs in unassigned)
    if len(assigned) != len(events) or len(set(assigned)) != len(assigned) or set(assigned) != set(events):
        raise ValueError("Executed fragments and unassigned events must cover every event exactly once")
    return {"schema": "cluster_tracks_v1", "tracks": tracks, "fragments": fragments,
            "trajectory_scope": trajectory_scope, "task_scope": copy.deepcopy(context.get("task_scope", {})),
            "unassigned": unassigned, "unresolved_links": unresolved, "graph_issues": graph_issues,
            "rule_config": rules, "rule_semantics": "not_applied_to_road_intents; 65_mps_is_diagnostic"
                if road_intents else "engineering_interpolation_heuristics_not_physical_guarantees",
            "identity_status": "unknown", "physical_status": "unknown", "correction_calls": 0,
            **({"planning_protocol": "road_intents_v1", "road_intent_settings": settings,
                "road_intent_semantics": "full_supplied_graph_route_intentions_and_conditional_time_interpolation",
                "route_audits": route_audits,
                "search_expansions": budget["expansions"], "search_truncated": budget["truncated"],
                "hard_time_guard_active": _cluster_hard_time_guard(settings["contract"]),
                "time_bound_pruned_edges": budget["time_bound_pruned_edges"],
                "fallback_applied": False} if road_intents else {})}
