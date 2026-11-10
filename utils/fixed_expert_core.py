"""Immutable snapshot identities and label-free retrieval implementation for fixed expert libraries."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import random
from pathlib import Path

MODES = ("none", "random", "structural_similarity")
CASE_FIELDS = ("case_id", "source_group_id", "signature", "failure_pattern",
               "expert_strategy", "camera_trace", "pattern_tags", "expert_summary", "expert_rules")


def digest(value):
    def canonical(obj):
        if isinstance(obj, dict):
            return {str(key): canonical(val) for key, val in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [canonical(val) for val in obj]
        return obj
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":"),
                                    default=lambda x: x.item()).encode()).hexdigest()


def source_group(record):
    """Use only explicitly defined provenance metadata; distinct case IDs do not imply distinct source vehicles."""
    for obj in (record, record.get("metadata", {}), record.get("meta", {})):
        group = obj.get("source_group_id")
        if group is not None and str(group).strip().lower() not in ("", "unknown", "none", "null"):
            return str(group)
        for field, prefix in (("vehicle_id", "vehicle"), ("trajectory_id", "trajectory")):
            if obj.get(field) is not None:
                return f"{prefix}:{obj[field]}"
    return "unknown"


def normalize_library(data):
    result = copy.deepcopy(data or {"cases": [], "global_rules": []})
    result.setdefault("global_rules", [])
    seen = set()
    for case in result.setdefault("cases", []):
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError("Snapshot requires unique nonempty case_id values")
        seen.add(case_id)
        case["case_id"] = case_id
        case["source_group_id"] = source_group(case)
        case.setdefault("failure_pattern", [])
        case.setdefault("expert_strategy", [])
        case.setdefault("camera_trace", [])
        case.setdefault("signature", {})
        case["content_hash"] = digest({k: case.get(k) for k in CASE_FIELDS})
    version = "fixed-" + digest({"rules": result["global_rules"],
                                  "cases": [c["content_hash"] for c in result["cases"]]})[:24]
    result["library_version"] = version
    for case in result["cases"]:
        case["library_version"] = version
    return result


def eligible_cases(cases, query_id=None, source_group_id=None, camera_trace=None):
    """Exclude self and same-source matches by case ID, source group, and observation trace without fabricating group links when provenance is unknown."""
    query_trace_hash = None if camera_trace is None else digest(camera_trace)
    return [case for case in cases
            if str(case["case_id"]) != str(query_id)
            and (not source_group_id or source_group_id == "unknown"
                 or case["source_group_id"] != str(source_group_id))
            and (camera_trace is None or digest(case["camera_trace"]) != query_trace_hash)]


_PROCESS_PRIVATE_FIELDS = {
    "ground_truth", "ground_truth_by_vehicle", "gt", "f1", "precision", "recall",
    "evaluation", "recovery_eval", "vehicle_id", "vehicle_ids", "record_labels",
    "utility", "train_action_effect", "failure_pattern", "baseline_score", "case_score",
}


def _process_public(value):
    if isinstance(value, dict):
        if any(key.lower() in _PROCESS_PRIVATE_FIELDS or "ground_truth" in key.lower()
               for key in value):
            raise ValueError("Process retrieval accepts public process evidence, not answers or labels")
        for item in value.values():
            _process_public(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _process_public(item)


def _process_known_source(group):
    return (isinstance(group, str) and group.strip().lower() not in
            ("", "unknown", "none", "null") and "unknown" not in group.lower())


def _process_observations(rows):
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not isinstance(row.get("obs_id"), str)
                   or not row["obs_id"] or type(row.get("camera_id")) is not int
                   or isinstance(row.get("time"), bool) or not isinstance(row.get("time"), (int, float))
                   or not math.isfinite(row["time"]) for row in rows)
            or len({row["obs_id"] for row in rows}) != len(rows)
            or any(a["time"] > b["time"] for a, b in zip(rows, rows[1:]))):
        raise ValueError("Process retrieval requires complete ordered typed public observations")
    return rows


def _local_unique_interval_mapping_record(case):
    """Retain explicit local TRAIN certificates without completing a chain."""
    audit = {"evidence_scope": "local_unique_intervals", "qualified_interval_count": 0,
             "excluded_interval_count": 0, "conflicting_cameras": [], "reason": None}
    source, group = case.get("source_id"), case.get("source_group_id")
    if (not ((type(source) is int and source >= 0) or _process_known_source(source))
            or not _process_known_source(group) or case.get("split", "train") != "train"):
        audit["reason"] = "unknown_source_or_non_train_annotation"
        return None, audit
    events, intervals = case.get("events"), case.get("intervals")
    conflicts = case.get("same_camera_node_conflict")
    if (not isinstance(events, list) or not events or not isinstance(intervals, list)
            or not isinstance(conflicts, list) or any(type(camera) is not int for camera in conflicts)
            or any(not isinstance(event, dict) or not isinstance(event.get("obs_id"), str)
                   or not event["obs_id"] or type(event.get("camera_id")) is not int
                   or isinstance(event.get("time"), bool)
                   or not isinstance(event.get("time"), (int, float))
                   or not math.isfinite(event["time"]) for event in events)
            or len({event["obs_id"] for event in events}) != len(events)
            or any(a["time"] > b["time"] for a, b in zip(events, events[1:]))):
        audit["reason"] = "invalid_original_observations_or_interval_list"
        return None, audit

    # Other locally supported occurrences can contradict an endpoint even if
    # their own intervals are absent. Never flatten such a camera to one node.
    camera_choices = {}
    for event in events:
        if (event.get("unique_occurrence_support") is True
                and event.get("association_status") == "conditional_training_window_support"
                and type(event.get("conditional_node")) is int):
            camera_choices.setdefault(event["camera_id"], set()).add(event["conditional_node"])
    conflicts = set(conflicts) | {camera for camera, nodes in camera_choices.items() if len(nodes) > 1}
    audit["conflicting_cameras"] = sorted(conflicts)
    positions = {event["obs_id"]: i for i, event in enumerate(events)}

    def certified(event):
        occurrences, nodes = event.get("supported_gt_occurrences"), event.get("supported_nodes")
        return (event.get("unique_occurrence_support") is True
                and event.get("association_status") == "conditional_training_window_support"
                and type(event.get("conditional_node")) is int
                and type(event.get("known_training_vehicle_source")) is type(source)
                and event.get("known_training_vehicle_source") == source
                and isinstance(occurrences, list) and len(occurrences) == 1
                and type(occurrences[0]) is int and occurrences[0] >= 0
                and isinstance(nodes, list) and len(nodes) == 1
                and type(nodes[0]) is int and nodes[0] == event["conditional_node"])

    mapping, pairs, endpoint_ids = {}, set(), set()
    for interval in intervals:
        qualified = False
        if isinstance(interval, dict):
            left_id, right_id = interval.get("from_obs_id"), interval.get("to_obs_id")
            if (isinstance(left_id, str) and isinstance(right_id, str)
                    and left_id in positions and right_id in positions
                    and positions[right_id] == positions[left_id] + 1):
                left, right = events[positions[left_id]], events[positions[right_id]]
                gap, road = interval.get("training_observed_time_gap_s"), interval.get("road_node_order")
                qualified = (certified(left) and certified(right)
                    and left["camera_id"] != right["camera_id"]
                    and left["camera_id"] not in conflicts and right["camera_id"] not in conflicts
                    and all(type(interval.get(key)) is int for key in
                            ("from_camera_id", "to_camera_id", "from_node", "to_node"))
                    and interval["from_camera_id"] == left["camera_id"]
                    and interval["to_camera_id"] == right["camera_id"]
                    and interval["from_node"] == left["conditional_node"]
                    and interval["to_node"] == right["conditional_node"]
                    and interval.get("alignment_status") == "conditional_unique_training_occurrence"
                    and type(interval.get("complete_unique_chain")) is bool
                    and interval.get("hard_physical_mapping_or_travel_time") is False
                    and not isinstance(gap, bool) and isinstance(gap, (int, float))
                    and math.isfinite(gap) and gap >= 0
                    and math.isclose(gap, right["time"] - left["time"], rel_tol=1e-9, abs_tol=1e-9)
                    and isinstance(road, list) and len(road) >= 2
                    and all(type(node) is int for node in road)
                    and road[0] == left["conditional_node"] and road[-1] == right["conditional_node"]
                    and right["supported_gt_occurrences"][0] > left["supported_gt_occurrences"][0]
                    and len(road) == right["supported_gt_occurrences"][0] - left["supported_gt_occurrences"][0] + 1)
                if qualified:
                    mapping[left["camera_id"]] = left["conditional_node"]
                    mapping[right["camera_id"]] = right["conditional_node"]
                    pairs.add((left["camera_id"], right["camera_id"],
                               left["conditional_node"], right["conditional_node"]))
                    endpoint_ids.update((left_id, right_id))
        audit["qualified_interval_count" if qualified else "excluded_interval_count"] += 1
    if not pairs:
        audit["reason"] = "no_certified_original_adjacent_interval"
        return None, audit

    # Soft neighbors retain original observations; filtering before adjacency
    # would invent a connection across an unknown or excluded middle event.
    neighbors = {camera: set() for camera in mapping}
    for i, event in enumerate(events):
        if event["obs_id"] in endpoint_ids:
            neighbors[event["camera_id"]].add((events[i - 1]["camera_id"] if i else None,
                events[i + 1]["camera_id"] if i + 1 < len(events) else None))
    return {"case_id": case["case_id"], "source_id": source, "source_group_id": group,
            "evidence_scope": "local_unique_intervals", "camera_nodes": mapping,
            "directed_adjacent_pairs": sorted(pairs),
            "historical_neighbor_cameras": {camera: sorted(values, key=lambda row: digest(row))
                                             for camera, values in neighbors.items()}}, audit


def build_mapping_experience_index(evidence_cases, *, include_local_unique_intervals=False):
    """Index conditional TRAIN annotations once, retaining no paths or times.

    Complete unique alignment is annotation support, not physical truth or an
    actual recovery process. Source eligibility/independence is supplied by the
    caller's original audit; different identifier strings never certify it.
    The optional extension only adds explicit local interval certificates from
    otherwise excluded parents; it does not complete or rewrite strict cases.
    """
    if type(include_local_unique_intervals) is not bool:
        raise ValueError("Local interval inclusion must be an explicit boolean")
    cases = list(evidence_cases.values()) if isinstance(evidence_cases, dict) else list(evidence_cases)
    index = {"schema": "conditional_train_mapping_index_v1", "cases": {}, "by_camera": {},
             "audit": {"new_API_requests": 0, "graph_searches": 0, "query_answers_used": False,
                       "trained_utility": False, "input_case_count": len(cases), "cases": []}}
    seen = set()
    for case in cases:
        cid = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Mapping evidence requires unique nonempty case IDs")
        seen.add(cid)
        decision = {"case_id": cid, "reasons": []}
        index["audit"]["cases"].append(decision)
        source, group = case.get("source_id"), case.get("source_group_id")
        known_source = (type(source) is int and source >= 0) or _process_known_source(source)
        if not known_source or not _process_known_source(group):
            decision["reasons"].append("unknown_source_id_or_group")
        if (case.get("extraction_eligible") is not True or case.get("full_chain_complete") is not True
                or case.get("full_chain_unique") is not True
                or type(case.get("full_chain_assignment_count_capped_at_two")) is not int
                or case["full_chain_assignment_count_capped_at_two"] != 1
                or case.get("split", "train") != "train"):
            decision["reasons"].append("not_complete_unique_train_annotation")
        if case.get("same_camera_node_conflict") != []:
            decision["reasons"].append("same_camera_node_conflict_or_unknown")
        events = case.get("events")
        if (not isinstance(events, list) or not events
                or any(not isinstance(event, dict) or not isinstance(event.get("obs_id"), str)
                       or not event["obs_id"] or type(event.get("camera_id")) is not int
                       or type(event.get("conditional_node")) is not int
                       or event.get("association_status") != "conditional_training_window_support"
                       or event.get("unique_occurrence_support") is not True
                       or isinstance(event.get("time"), bool)
                       or not isinstance(event.get("time"), (int, float))
                       or not math.isfinite(event["time"]) for event in events)
                or len({event["obs_id"] for event in events}) != len(events)
                or any(a["time"] > b["time"] for a, b in zip(events, events[1:]))):
            decision["reasons"].append("incomplete_typed_conditional_events")
        if decision["reasons"]:
            continue
        mapping = {}
        for event in events:
            camera, node = event["camera_id"], event["conditional_node"]
            if camera in mapping and mapping[camera] != node:
                decision["reasons"].append("inconsistent_same_camera_annotation")
                break
            mapping[camera] = node
        if decision["reasons"]:
            continue
        # Neighbors describe historical camera context, never an absolute time
        # or full road-node trajectory. Repeated runs keep one camera mapping.
        order = [event["camera_id"] for event in events]
        runs = [camera for i, camera in enumerate(order) if i == 0 or camera != order[i - 1]]
        neighbors = {camera: set() for camera in mapping}
        for i, camera in enumerate(runs):
            neighbors[camera].add((runs[i - 1] if i else None,
                                   runs[i + 1] if i + 1 < len(runs) else None))
        record = {"case_id": cid, "source_id": source, "source_group_id": group,
                  "camera_nodes": mapping,
                  "directed_adjacent_pairs": sorted({(a["camera_id"], b["camera_id"],
                      a["conditional_node"], b["conditional_node"]) for a, b in zip(events, events[1:])
                      if a["camera_id"] != b["camera_id"]}),
                  "historical_neighbor_cameras": {camera: sorted(values, key=lambda row: digest(row))
                                                   for camera, values in neighbors.items()}}
        index["cases"][cid] = record
        for camera in mapping:
            index["by_camera"].setdefault(camera, []).append(cid)
    if include_local_unique_intervals:
        index["audit"].update(extension_scope="local_unique_intervals",
            strict_complete_unique_case_count=len(index["cases"]), local_interval_case_count=0)
        for case, decision in zip(cases, index["audit"]["cases"]):
            cid = case["case_id"]
            if cid in index["cases"]:
                continue
            record, local_audit = _local_unique_interval_mapping_record(case)
            decision["local_interval_extension"] = local_audit
            if record is None:
                continue
            decision["indexed_evidence_scope"] = "local_unique_intervals"
            index["cases"][cid] = record
            index["audit"]["local_interval_case_count"] += 1
            for camera in record["camera_nodes"]:
                index["by_camera"].setdefault(camera, []).append(cid)
    for ids in index["by_camera"].values():
        ids.sort()
    index["audit"]["indexed_case_count"] = len(index["cases"])
    return index


def retrieve_mapping_experience(context, index, *, source_eligible_ids, candidate_rows,
                                selection="structural", seed=0, max_characters=2400):
    """Retrieve one soft mapping teacher from the caller's audited shared pool.

    There are no API calls, graph searches, labels, score-based query selection
    or trained utility. Camera-only support is disjoint additional support; a
    reverse pair cannot vote for current direction. Source ID OR source group
    equality merges votes, even when multiple training cases are present.
    """
    from utils.fixed_expert_recovery import (_camera_mapping_public_context,
                                              format_mapping_experience_context)

    _camera_mapping_public_context(context)
    _process_public(context)
    _process_public(candidate_rows)
    if selection not in {"structural", "random"}:
        raise ValueError("Mapping selection must be structural or random")
    if type(max_characters) is not int or max_characters < 1 or type(seed) is not int:
        raise ValueError("Mapping retrieval requires a positive character budget and integer seed")
    if not isinstance(index, dict) or index.get("schema") != "conditional_train_mapping_index_v1":
        raise ValueError("Mapping retrieval requires the conditional TRAIN index")
    observations = _process_observations(context.get("observations"))
    candidates, current_ids = {}, {}
    for row in context.get("camera_mapping_candidates", []):
        camera = row.get("camera_id")
        nodes = [item.get("node") for item in row.get("candidates", [])]
        if (type(camera) is not int or camera in candidates or len(nodes) != 2
                or any(type(node) is not int for node in nodes) or len(set(nodes)) != 2):
            raise ValueError("Mapping retrieval requires unchanged unique typed Top2 candidates")
        candidates[camera] = nodes
    for row in observations:
        if row["camera_id"] not in candidates:
            raise ValueError("Every current camera needs its original Top2 candidates")
        current_ids.setdefault(row["camera_id"], []).append(row["obs_id"])
    if set(candidates) != set(current_ids):
        raise ValueError("Mapping candidates must cover exactly the current observed cameras")
    adjacent = {}
    for left, right in zip(observations, observations[1:]):
        adjacent.setdefault((left["camera_id"], right["camera_id"]), []).append(
            [left["obs_id"], right["obs_id"]])
    allowed, seen, pool = set(source_eligible_ids), set(), []
    decisions = []
    for row in candidate_rows:
        cid, score = row.get("case_id"), row.get("structural_score")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Mapping candidate rows require unique nonempty case IDs")
        seen.add(cid)
        decision = {"case_id": cid}
        decisions.append(decision)
        if cid not in allowed:
            decision["reason"] = "not_in_caller_source_audit"
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Eligible mapping candidates require finite structural_score")
        record = index["cases"].get(cid)
        if record is None:
            decision["reason"] = "no_complete_unique_train_annotation"
            continue
        shared = {camera: node for camera, node in record["camera_nodes"].items()
                  if camera in candidates and node in candidates[camera]}
        if not shared:
            decision["reason"] = "no_shared_current_top2_mapping"
            continue
        pairs = {tuple(pair) for pair in record["directed_adjacent_pairs"]
                 if tuple(pair[:2]) in adjacent and pair[2] in candidates[pair[0]]
                 and pair[3] in candidates[pair[1]]}
        covered = sum(len(ids) for cameras, ids in adjacent.items()
                      if any(pair[:2] == cameras for pair in pairs))
        item = {"record": record, "case_id": cid, "shared": shared, "pairs": pairs,
                "complete_current_pair_coverage": covered, "shared_camera_coverage": len(shared),
                "structural_score": float(score)}
        pool.append(item)
        decision.update(reason="eligible_conditional_mapping", complete_current_pair_coverage=covered,
                        shared_camera_coverage=len(shared))
    pool.sort(key=lambda row: (-row["complete_current_pair_coverage"], -row["shared_camera_coverage"],
                              -row["structural_score"], row["case_id"]))
    audit = {"schema": "mapping_experience_retrieval_audit_v1", "selection": selection,
             "pool_ids": [row["case_id"] for row in pool], "decisions": decisions,
             "new_API_requests": 0, "graph_searches": 0, "trained_utility": False,
             "query_answers_used": False, "support_source_guard": "caller_supplied_original_source_audit",
             "support_units": "known_source_id_or_group_connected_components_in_same_safe_pool",
             "selected_case_id": None, "characters": 0, "omitted_record_count": 0,
             "selection_policy": "complete same-direction current adjacent pair coverage, shared Top2 camera coverage, structural_score, case_id"}
    if not pool:
        audit["reason"] = "no_source_safe_shared_mapping"
        return {"payload": None, "audit": audit}
    # Deduplicate known source identities conservatively. These equalities only
    # merge votes; they do not establish query/reference independence.
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    for item in pool:
        record = item["record"]
        source = ("source_id", type(record["source_id"]).__name__, record["source_id"])
        group = ("source_group", record["source_group_id"])
        parents[find(source)] = find(group)
        item["source_key"] = source
    camera_sources, direction_sources, pair_sources = {}, {}, {}
    for item in pool:
        component = find(item["source_key"])
        for camera, node in item["shared"].items():
            camera_sources.setdefault((camera, node), set()).add(component)
        for a, b, left, right in item["pairs"]:
            pair_sources.setdefault((a, b, left, right), set()).add(component)
            direction_sources.setdefault((a, left), set()).add(component)
            direction_sources.setdefault((b, right), set()).add(component)
    selected = pool[0]
    if selection == "random":
        shuffled = sorted(pool, key=lambda row: row["case_id"])
        random.Random(digest([seed, [[row["camera_id"], row["time"]] for row in observations]])).shuffle(shuffled)
        selected = shuffled[0]
        audit["selection_policy"] = "fixed seed-hash random teacher from exactly the same qualified pool"
    camera_rows, pair_rows = [], []
    for camera in sorted(candidates):
        supports, sources = [], []
        for node in candidates[camera]:
            all_sources = camera_sources.get((camera, node), set())
            directed = direction_sources.get((camera, node), set())
            supports.append([node, len(directed), len(all_sources - directed),
                             "conditional_train_annotation" if all_sources else "unknown"])
            sources.append(all_sources)
        camera_rows.append([camera, current_ids[camera], supports,
                            sum(bool(values) for values in sources) > 1,
                            bool(sources[0] & sources[1])])
    for (a, b), ids in adjacent.items():
        rows = [[left, right, len(values)] for (ca, cb, left, right), values in sorted(pair_sources.items())
                if (ca, cb) == (a, b)]
        values = [pair_sources[(a, b, left, right)] for left, right, _ in rows]
        source_conflict = any(x & y for i, x in enumerate(values) for y in values[i + 1:])
        pair_rows.append([ids, [a, b], rows, len(rows) > 1, source_conflict])
    teacher = selected["record"]
    payload = {"schema": "conditional_train_mapping_experience_v1",
        "semantics": "Soft conditional TRAIN annotations; whole-chain uniqueness is not physical truth. "
            "Use mapping support in camera_node_choices alongside current direction and path comparison; do not force-copy "
            "the teacher, add/remove candidates or adjust times. Same-direction counts require both Top2 endpoints of an "
            "actual historical adjacent directed camera pair; camera-only counts are additional sources with unknown direction. "
            "Known-source counts use the same caller-audited pool, not probability or utility. All pool conflicts are shown. "
            "Absent support and unlisted pair combinations are unknown. A matching directed interval is more applicable "
            "than camera-only recurrence; resolve conflicts using the full current observation chain and pair facts. "
            "Output the chosen mapping in camera_node_choices before generating interval intents.",
        "camera_support_columns": ["camera_id", "current_obs_ids", "node_support", "pool_node_conflict", "pool_source_conflict"],
        "node_support_columns": ["node", "same_direction_sources", "camera_only_sources", "status"],
        "camera_support": camera_rows,
        "pair_support_columns": ["current_obs_id_pairs", "camera_ids", "node_pair_source_counts", "pool_pair_conflict", "pool_source_conflict"],
        "pair_support": pair_rows,
        "teacher": {"case_id": selected["case_id"],
            "mapping_columns": ["camera_id", "node", "historical_neighbor_cameras"],
            "mapping": [[camera, node, teacher["historical_neighbor_cameras"][camera]]
                        for camera, node in sorted(selected["shared"].items())]}}
    block = format_mapping_experience_context(payload)
    audit.update(selected_case_id=selected["case_id"], complete_record_characters=len(block),
                 complete_current_pair_coverage=selected["complete_current_pair_coverage"],
                 shared_camera_coverage=selected["shared_camera_coverage"],
                 known_distinct_sources=len({find(item["source_key"]) for item in pool}),
                 pool_camera_conflicts=[row[0] for row in camera_rows if row[3]],
                 pool_camera_source_conflicts=[row[0] for row in camera_rows if row[4]],
                 pool_pair_conflicts=[row[1] for row in pair_rows if row[3]],
                 pool_pair_source_conflicts=[row[1] for row in pair_rows if row[4]])
    if len(block) > max_characters:
        audit.update(reason="whole_mapping_record_exceeds_budget", omitted_record_count=1)
        return {"payload": None, "audit": audit}
    audit.update(reason=None, characters=len(block))
    return {"payload": payload, "audit": audit}


def build_teacher_interval_kinematics_index(evidence_cases):
    """Index pass-through vs. turn-back teacher interval durations from TRAIN evidence."""
    if not isinstance(evidence_cases, list):
        raise ValueError("Teacher kinematics index requires a list of training evidence cases")
    index = {}
    for case in evidence_cases:
        if not isinstance(case, dict) or case.get("split", "train") != "train" or not case.get("extraction_eligible"):
            continue
        events = [e for e in case.get("events", []) if isinstance(e, dict) and "obs_id" in e]
        pos = {e["obs_id"]: i for i, e in enumerate(events)}
        for iv in case.get("intervals", []):
            key = tuple(iv.get(k) for k in ("from_camera_id", "to_camera_id", "from_node", "to_node"))
            gap = iv.get("training_observed_time_gap_s")
            if any(type(v) is not int for v in key) or isinstance(gap, bool) or not isinstance(gap, numbers.Real) or gap <= 0:
                continue
            idx = pos.get(iv.get("to_obs_id"), -1)
            turns_back = (0 <= idx < len(events) - 1 and events[idx + 1].get("camera_id") == key[0])
            entry = index.setdefault(key, {"all_gaps_s": [], "pass_through_gaps_s": [], "turn_back_gaps_s": []})
            entry["all_gaps_s"].append(float(gap))
            entry["turn_back_gaps_s" if turns_back else "pass_through_gaps_s"].append(float(gap))
    return index


def evaluate_g5_kinematic_boundary_gate(context, g5_gate_audit, *, road_graph,
                                        teacher_kinematics_index, v_stop_mps=5.0):
    """Apply kinematic boundary consistency check G5^kin on top of a G5 gate audit."""
    import networkx as nx
    if road_graph is None or not isinstance(teacher_kinematics_index, dict):
        raise ValueError("g5_kinematic_boundary_v1 requires road_graph and teacher_kinematics_index")
    if g5_gate_audit is None:
        return {"schema": "g5_any_local_gate_v1", "policy": "g5_kinematic_boundary_v1",
                "v_stop_mps": float(v_stop_mps), "allow": True,
                "reason": "strict_match_short_circuited_local_gate", "bindings": [], "graph_searches": 0}
    observations = _process_observations(context.get("observations"))
    obs_time = {r["obs_id"]: float(r["time"]) for r in observations}
    last_oid = observations[-1]["obs_id"]
    cands = {r["camera_id"]: [c["node"] for c in r["candidates"]]
             for r in context.get("camera_mapping_candidates", []) if isinstance(r, dict)}
    audit = {**copy.deepcopy(g5_gate_audit), "policy": "g5_kinematic_boundary_v1", "v_stop_mps": float(v_stop_mps)}
    searches = 0
    for b in audit.get("bindings", []):
        if not b.get("passes"):
            b.update(touches_terminal_dest=False, kinematic_consistent=False)
            continue
        term_pairs = [p for p in b.get("current_obs_id_pairs", []) if len(p) == 2 and p[1] == last_oid]
        cam_a, cam_b = b["directed_camera_pair"]
        if not (b.get("touches_boundary") and term_pairs and b["camera_id"] == cam_b and cands.get(cam_a)):
            b.update(touches_terminal_dest=False, kinematic_consistent=True)
            continue
        u_a, prior_n, supp_n = cands[cam_a][0], b["public_prior_node"], b["conditional_supported_node"]
        try:
            searches += 2
            d_supp = float(nx.shortest_path_length(road_graph, u_a, supp_n, weight="length"))
            d_prior = float(nx.shortest_path_length(road_graph, u_a, prior_n, weight="length"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            b.update(touches_terminal_dest=True, kinematic_consistent=False, passes=False)
            continue
        pass_gaps = teacher_kinematics_index.get((cam_a, cam_b, u_a, supp_n), {}).get("pass_through_gaps_s", [])
        dt_q = min((obs_time[p[1]] - obs_time[p[0]] for p in term_pairs if obs_time[p[1]] > obs_time[p[0]]), default=0.0)
        v_trunc = (d_supp / dt_q) if dt_q > 0 else float("inf")
        trunc_m = bool(d_supp < d_prior and pass_gaps)
        ok = True if not trunc_m else (dt_q > max(pass_gaps) and v_trunc <= float(v_stop_mps))
        b.update(touches_terminal_dest=True, trunc_m=trunc_m, kinematic_consistent=ok, passes=ok,
                 d_supp_m=round(d_supp, 3), d_prior_m=round(d_prior, 3), query_terminal_dt_s=round(dt_q, 3))
    audit["graph_searches"] = int(audit.get("graph_searches", 0)) + searches
    audit["allow"] = any(bool(b.get("passes")) for b in audit.get("bindings", []))
    audit["reason"] = "at_least_one_binding_passes_kinematic_gate" if audit["allow"] else "all_local_bindings_fail_g5_kinematic_boundary_v1"
    return audit


def _filter_payload_by_passing_bindings(payload, bindings):
    """Prune non-passing trigger bindings elementwise while preserving typed support table invariants."""
    if not isinstance(payload, dict) or not bindings:
        return payload
    passing_pairs = {
        (int(b["camera_id"]), int(b["conditional_supported_node"]))
        for b in bindings
        if b.get("passes") and "camera_id" in b and "conditional_supported_node" in b
    }
    rejected_pairs = {
        (int(b["camera_id"]), int(b["conditional_supported_node"]))
        for b in bindings
        if not b.get("passes") and "camera_id" in b and "conditional_supported_node" in b
    } - passing_pairs
    if not rejected_pairs:
        return payload
    filtered = copy.deepcopy(payload)
    node_pair_counts = {}
    new_pair_support = []
    for row in filtered.get("pair_support", []):
        obs_pairs, cam_pair, combos, _, src_conflict = row
        a, b = int(cam_pair[0]), int(cam_pair[1])
        kept = [
            list(item)
            for item in combos
            if (a, int(item[0])) not in rejected_pairs and (b, int(item[1])) not in rejected_pairs
        ]
        for left, right, cnt in kept:
            if cnt:
                node_pair_counts.setdefault((a, int(left)), []).append(int(cnt))
                node_pair_counts.setdefault((b, int(right)), []).append(int(cnt))
        new_pair_support.append([obs_pairs, cam_pair, kept, len(kept) > 1, bool(src_conflict and len(kept) > 1)])
    filtered["pair_support"] = new_pair_support

    new_cam_support = []
    for row in filtered.get("camera_support", []):
        cam, obs_ids, supports, _, src_conflict = row
        cam_id = int(cam)
        new_supports = []
        for node, same_dir, cam_only, _ in supports:
            node_id = int(node)
            if (cam_id, node_id) in rejected_pairs:
                new_supports.append([node_id, 0, 0, "unknown"])
            else:
                counts = node_pair_counts.get((cam_id, node_id), [])
                lo, hi = max(counts, default=0), sum(counts)
                adj_same_dir = min(max(int(same_dir), lo), hi) if counts else 0
                adj_cam_only = int(cam_only)
                status = "conditional_train_annotation" if (adj_same_dir + adj_cam_only) > 0 else "unknown"
                new_supports.append([node_id, adj_same_dir, adj_cam_only, status])
        active_nodes = sum(1 for item in new_supports if (item[1] + item[2]) > 0)
        new_cam_support.append([cam, obs_ids, new_supports, active_nodes > 1, bool(src_conflict and active_nodes > 1)])
    filtered["camera_support"] = new_cam_support

    teacher = filtered.get("teacher")
    if isinstance(teacher, dict) and isinstance(teacher.get("mapping"), list):
        teacher["mapping"] = [
            list(item)
            for item in teacher["mapping"]
            if (int(item[0]), int(item[1])) not in rejected_pairs
        ]
    return filtered


def retrieve_mapping_experience_with_local_fallback(context, strict_index, local_index, *,
        source_eligible_ids, candidate_rows, selection="structural", seed=0, max_characters=4000,
        local_gate="g5_kinematic_boundary_v1", road_graph=None, teacher_kinematics_index=None,
        v_stop_mps=5.0, exclude_source_group=None):
    """Retrieve whole-chain directional matches first and apply elementwise G5/G5^kin filtering on local fallback."""
    if local_gate not in (None, "g5_any", "g5_any_time_boundary_v1", "g5_kinematic_boundary_v1"):
        raise ValueError("Local mapping gate must be None or g5_any or g5_any_time_boundary_v1 or g5_kinematic_boundary_v1")
    if (not isinstance(strict_index, dict) or not isinstance(local_index, dict)
            or strict_index.get("schema") != "conditional_train_mapping_index_v1"
            or local_index.get("schema") != strict_index.get("schema")
            or not isinstance(strict_index.get("cases"), dict)
            or not isinstance(local_index.get("cases"), dict)
            or any(local_index["cases"].get(cid) != record
                   for cid, record in strict_index["cases"].items())):
        raise ValueError("Local fallback index must preserve every strict mapping record")
    source_eligible_ids, candidate_rows = list(source_eligible_ids), list(candidate_rows)
    if exclude_source_group is not None and str(exclude_source_group).strip():
        excl = str(exclude_source_group).strip()
        disallowed_cids = {
            cid for cid, rec in local_index["cases"].items()
            if isinstance(rec, dict) and str(
                rec.get("source_group_id") or rec.get("source_group") or rec.get("group_id") or rec.get("source_id") or ""
            ) == excl
        }
        for row in candidate_rows:
            if isinstance(row, dict) and str(
                row.get("source_group_id") or row.get("source_group") or row.get("group_id") or row.get("source_id") or ""
            ) == excl:
                if "case_id" in row:
                    disallowed_cids.add(row["case_id"])
        source_eligible_ids = [cid for cid in source_eligible_ids if cid not in disallowed_cids]
        candidate_rows = [
            row for row in candidate_rows
            if not (isinstance(row, dict) and row.get("case_id") in disallowed_cids)
        ]
    options = {"source_eligible_ids": source_eligible_ids, "candidate_rows": candidate_rows,
               "selection": selection, "seed": seed, "max_characters": max_characters}
    strict = retrieve_mapping_experience(context, strict_index, **options)
    strict_gate = gate_mapping_experience(context, strict["payload"])
    if strict_gate["inject"]:
        return {**strict, "gate": strict_gate, "strict_gate": strict_gate["audit"],
                "fallback_used": False, "selection_scope": "strict"}
    local = retrieve_mapping_experience(context, local_index, **options)
    local_result = {**local, "gate": gate_mapping_experience(context, local["payload"]),
                    "strict_gate": strict_gate["audit"], "fallback_used": True,
                    "selection_scope": "local_unique_intervals_fallback"}
    if local_gate is None:
        return local_result
    time_boundary_policy = local_gate in ("g5_any_time_boundary_v1", "g5_kinematic_boundary_v1")
    audit = {"schema": "g5_elementwise_local_gate_v1", "policy": local_gate,
             "formula": "elementwise filter {b : (not b.touches_boundary) or b.known_source_count >= 2}",
             "boundary_definition": ("validated_context_time_order_first_last" if time_boundary_policy
                                     else "numeric_obs_id_suffix_min_max_from_complete_camera_support"),
             "query_answers_used": False,
             "new_API_requests": 0, "graph_searches": 0, "bindings": [], "allow": False}
    if not local_result["gate"]["inject"]:
        audit["reason"] = "original_local_gate_abstained"
        return {**local_result, "g5_gate": audit, "local_declined": False}
    support = {row[0]: row[1] for row in local["payload"]["camera_support"]}
    obs_ids = {oid for ids in support.values() for oid in ids}
    try:
        suffixes = {oid: int(oid.rsplit(":", 1)[-1]) for oid in obs_ids}
    except (ValueError, AttributeError):
        raise ValueError("g5_any requires numeric original observation ID suffixes") from None
    if len(set(suffixes.values())) != len(suffixes):
        raise ValueError("g5_any requires unambiguous numeric observation IDs")
    ordered = sorted(obs_ids, key=suffixes.get)
    raw_boundary = [ordered[0], ordered[-1]]
    observations = _process_observations(context["observations"])
    context_boundary = [observations[0]["obs_id"], observations[-1]["obs_id"]]
    boundary = context_boundary if time_boundary_policy else raw_boundary
    cameras = {row["obs_id"]: row["camera_id"] for row in observations}
    audit.update(boundary_obs_ids=boundary, context_boundary_obs_ids=context_boundary,
                 boundary_matches_context_order=raw_boundary == context_boundary,
                 boundary_camera_ids=[cameras[oid] for oid in boundary])
    if time_boundary_policy:
        audit.update(raw_record_boundary_obs_ids=raw_boundary, time_order_reliable=True,
                     time_order_basis="_process_observations validates finite nondecreasing time and preserves stable input order")
    for binding in local_result["gate"]["audit"]["trigger_bindings"]:
        ids = support[binding["camera_id"]]
        touches = any(oid in boundary for oid in ids)
        audit["bindings"].append({**copy.deepcopy(binding), "obs_ids": list(ids),
            "touches_boundary": touches,
            "passes": not touches or binding["known_source_count"] >= 2})
    audit["allow"] = any(binding["passes"] for binding in audit["bindings"])
    audit["reason"] = "at_least_one_binding_passes" if audit["allow"] else "all_local_bindings_fail_g5"
    if local_gate == "g5_kinematic_boundary_v1" and audit["allow"]:
        audit = evaluate_g5_kinematic_boundary_gate(
            context,
            audit,
            road_graph=road_graph,
            teacher_kinematics_index=teacher_kinematics_index,
            v_stop_mps=v_stop_mps,
        )
    if audit["allow"]:
        from utils.fixed_expert_recovery import format_mapping_experience_context
        filtered_payload = _filter_payload_by_passing_bindings(local["payload"], audit["bindings"])
        filtered_gate = gate_mapping_experience(context, filtered_payload)
        if filtered_gate["inject"]:
            filtered_block = format_mapping_experience_context(filtered_payload)
            filtered_audit = {**copy.deepcopy(local["audit"]), "characters": len(filtered_block)}
            return {**local_result, "payload": filtered_payload, "audit": filtered_audit,
                    "gate": filtered_gate, "g5_gate": audit, "local_declined": False}
        audit["allow"] = False
        audit["reason"] = "no_passing_trigger_bindings_after_elementwise_filter"
    return {**strict, "gate": strict_gate, "strict_gate": strict_gate["audit"],
            "fallback_used": True, "selection_scope": "strict_after_g5_local_decline",
            "g5_gate": audit, "local_declined": True,
            "local_candidate_audit": local["audit"], "local_candidate_gate": local_result["gate"]["audit"]}


def gate_mapping_experience(context, payload):
    """Choose whether to inject the unchanged mapping block using public direction.

    This is a query-level soft-evidence gate, not a mapping decision, negative
    label, physical guarantee or trained utility. Missing support stays unknown.
    Teacher identity, labels, paths and results do not enter the gate rule.
    """
    from utils.fixed_expert_recovery import _camera_mapping_public_context

    _camera_mapping_public_context(context)
    _process_public(context)
    _process_public(payload)
    observations = _process_observations(context.get("observations"))
    candidates, current_ids, adjacent = {}, {}, {}
    for row in context.get("camera_mapping_candidates", []):
        if not isinstance(row, dict) or not isinstance(row.get("candidates"), list):
            raise ValueError("Direction gate requires original typed Top2 candidates")
        if any(not isinstance(item, dict) for item in row["candidates"]):
            raise ValueError("Direction gate requires original typed Top2 candidates")
        camera, nodes = row.get("camera_id"), [item.get("node") for item in row["candidates"]]
        if (type(camera) is not int or camera in candidates or len(nodes) != 2
                or any(type(node) is not int for node in nodes) or len(set(nodes)) != 2):
            raise ValueError("Direction gate requires original typed Top2 candidates")
        candidates[camera] = nodes
    for row in observations:
        current_ids.setdefault(row["camera_id"], []).append(row["obs_id"])
    if set(candidates) != set(current_ids):
        raise ValueError("Direction gate candidates must cover exactly the current cameras")
    for left, right in zip(observations, observations[1:]):
        adjacent.setdefault((left["camera_id"], right["camera_id"]), []).append(
            [left["obs_id"], right["obs_id"]])
    audit = {"schema": "direction_mapping_gate_audit_v1", "new_API_requests": 0, "graph_searches": 0,
             "trained_utility": False, "query_answers_used": False, "unsupported_is_unknown": True,
             "physical_proof_claimed": False, "candidates_changed": False, "trigger_bindings": []}
    if payload is None:
        audit["reason"] = "no_mapping_payload"
        return {"inject": False, "audit": audit}
    if (not isinstance(payload, dict) or payload.get("schema") != "conditional_train_mapping_experience_v1"
            or payload.get("camera_support_columns") != ["camera_id", "current_obs_ids", "node_support",
                                                          "pool_node_conflict", "pool_source_conflict"]
            or payload.get("node_support_columns") != ["node", "same_direction_sources", "camera_only_sources", "status"]
            or payload.get("pair_support_columns") != ["current_obs_id_pairs", "camera_ids", "node_pair_source_counts",
                                                        "pool_pair_conflict", "pool_source_conflict"]
            or not isinstance(payload.get("camera_support"), list)
            or not isinstance(payload.get("pair_support"), list)):
        raise ValueError("Direction gate requires the complete typed mapping support tables")
    cameras = {}
    for row in payload["camera_support"]:
        if (not isinstance(row, list) or len(row) != 5 or type(row[0]) is not int
                or row[0] not in candidates or row[0] in cameras or row[1] != current_ids[row[0]]
                or not isinstance(row[2], list) or len(row[2]) != 2
                or any(type(flag) is not bool for flag in row[3:5])):
            raise ValueError("Direction gate requires complete current observation bindings")
        nodes = {}
        for item in row[2]:
            if (not isinstance(item, list) or len(item) != 4 or type(item[0]) is not int
                    or item[0] not in candidates[row[0]] or item[0] in nodes
                    or any(type(count) is not int or count < 0 for count in item[1:3])
                    or item[3] != ("conditional_train_annotation" if item[1] + item[2] else "unknown")):
                raise ValueError("Direction gate requires Top2 nodes and nonnegative integer support counts")
            nodes[item[0]] = item
        cameras[row[0]] = {"nodes": nodes, "source_conflict": row[4]}
    if set(cameras) != set(candidates):
        raise ValueError("Direction gate requires every camera support row")
    pairs, node_pair_counts = {}, {}
    for row in payload["pair_support"]:
        if (not isinstance(row, list) or len(row) != 5 or not isinstance(row[1], list) or len(row[1]) != 2
                or any(type(camera) is not int for camera in row[1])
                or tuple(row[1]) not in adjacent or tuple(row[1]) in pairs
                or row[0] != adjacent[tuple(row[1])] or not isinstance(row[2], list)
                or any(type(flag) is not bool for flag in row[3:5])):
            raise ValueError("Direction gate requires complete original adjacent directed observation bindings")
        a, b = row[1]
        seen = set()
        for item in row[2]:
            if (not isinstance(item, list) or len(item) != 3
                    or any(type(value) is not int for value in item) or item[2] < 0
                    or item[0] not in candidates[a] or item[1] not in candidates[b]
                    or tuple(item[:2]) in seen or (a == b and item[2] > 0)):
                raise ValueError("Direction gate requires typed nonrepeated Top2 directional combinations")
            seen.add(tuple(item[:2]))
            if item[2]:
                for camera, node in ((a, item[0]), (b, item[1])):
                    node_pair_counts.setdefault((camera, node), []).append(item[2])
        pairs[(a, b)] = row
    if set(pairs) != set(adjacent):
        raise ValueError("Direction gate requires every original adjacent pair support row")
    for camera, row in cameras.items():
        for node, item in row["nodes"].items():
            counts = node_pair_counts.get((camera, node), [])
            if not max(counts, default=0) <= item[1] <= sum(counts):
                raise ValueError("Direction support counts must be grounded in current adjacent pair combinations")

    # The gate does not use camera-only counts or the teacher's case identity.
    # Zero support means unknown; it only prevents that support from triggering.
    for (a, b), row in pairs.items():
        if a == b or row[4] or cameras[a]["source_conflict"] or cameras[b]["source_conflict"]:
            continue
        for left, right, count in row[2]:
            if count <= 0:
                continue
            for camera, node in ((a, left), (b, right)):
                prior = candidates[camera][0]
                if node != prior and cameras[camera]["nodes"][prior][1] == 0:
                    audit["trigger_bindings"].append({"camera_id": camera, "public_prior_node": prior,
                        "conditional_supported_node": node, "current_obs_id_pairs": copy.deepcopy(row[0]),
                        "directed_camera_pair": [a, b], "known_source_count": count})
    inject = bool(audit["trigger_bindings"])
    audit["reason"] = "supported_nonprior_current_direction" if inject else "no_conflict_free_nonprior_direction_support"
    return {"inject": inject, "audit": audit}
