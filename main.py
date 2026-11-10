"""Main pipeline entrypoint for road-network trajectory recovery."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import pickle
from typing import Any, Dict

import numpy as np

from common import complete_trajectories_fast, merge_tm_adj_points, preprocess_paths_with_length
from eval import eval_trajs, evaluate, gen_vid2cid
from load_data import load_car_feat
from multi_llm_router import PerformanceAwareMultiLLMRouter, build_case_features_from_trace
from traj_rec_st_cot import LLMClient, build_cluster_track_context
from utils.fixed_expert_core import retrieve_mapping_experience_with_local_fallback
from utils.fixed_expert_recovery import (
    build_camera_mapping_context,
    build_camera_mapping_prompt,
    build_cluster_corridor_subgraph,
    evaluate_cluster_invocation_gate,
    execute_three_tier_corridor_verification,
    extract_kinematic_dp_corridors,
    parse_camera_mapping_response,
)
from utils.yaml_config_hook import add_config_arguments, flatten_dict, unflatten_dict, yaml_config_hook


def _build_top2_camera_candidates(road_graph, cam_xy_dict, camera_ids):
    """Compute geometric Top-2 candidate road nodes Gamma_2(c_i) per camera (Section II-A, Section III-C2)."""
    if not isinstance(cam_xy_dict, dict) or road_graph is None:
        return None
    node_xy = [
        (int(node), float(attr["x"]), float(attr["y"]))
        for node, attr in road_graph.nodes(data=True)
        if "x" in attr and "y" in attr
    ]
    if not node_xy:
        return None
    candidates = {}
    for cam in sorted({int(c) for c in camera_ids}):
        xy = cam_xy_dict.get(cam, cam_xy_dict.get(str(cam)))
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        cx, cy = float(xy[0]), float(xy[1])
        ranked = sorted((math.hypot(cx - nx_val, cy - ny_val), node) for node, nx_val, ny_val in node_xy)[:2]
        if ranked:
            candidates[cam] = [node for _, node in ranked]
    return candidates or None


def _extract_bank_candidate_bindings(active_mapping, active_experience):
    """Extract multi-candidate bindings B(O_q) when retrieved bank evidence differs from Top-1."""
    if not isinstance(active_experience, dict):
        return None
    bindings = {}
    for row in active_experience.get("camera_support") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            cam_id = int(row[0])
            cands = [int(active_mapping[cam_id])] if cam_id in active_mapping else []
            for ns in row[2] or []:
                if isinstance(ns, (list, tuple)) and ns:
                    cands.append(int(ns[0]))
            if len(set(cands)) > 1:
                bindings[cam_id] = sorted(set(cands))
    for row in (active_experience.get("teacher") or {}).get("mapping") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            cam_id, t_node = int(row[0]), int(row[1])
            cands = set(bindings.get(cam_id, []))
            if cam_id in active_mapping:
                cands.add(int(active_mapping[cam_id]))
            cands.add(t_node)
            if len(cands) > 1:
                bindings[cam_id] = sorted(cands)
    return bindings or None


def _build_public_mapping_context(camera_trace, *, road_graph, camera_to_node, cam_xy_dict):
    """Construct the public camera_mapping_v1 context from an ordered camera trace."""
    if road_graph is None or road_graph.number_of_nodes() < 2 or not isinstance(cam_xy_dict, dict):
        return None
    trace, points, provenance, positions = [], [], [], {}
    for idx, (cam, stamp) in enumerate(camera_trace):
        cam_id, ts = int(cam), float(stamp)
        xy_raw = cam_xy_dict.get(cam_id, cam_xy_dict.get(str(cam_id)))
        if not isinstance(xy_raw, (list, tuple)) or len(xy_raw) < 2:
            return None
        xy = [float(xy_raw[0]), float(xy_raw[1])]
        trace.append([cam_id, ts])
        points.append([xy[0], xy[1], ts])
        positions[str(cam_id)] = xy
        provenance.append({
            "input_index": idx,
            "camera_id": cam_id,
            "prepared_time": ts,
            "raw_record_indices": [idx],
            "transformed_times": [ts],
            "raw_times_s": [ts],
            "aggregation": {"exact_reproduction": True, "method": "one_raw_record_event"},
        })
    if not trace:
        return None
    query = {
        "camera_trace": trace,
        "observed_points": points,
        "observation_provenance": provenance,
        "camera_positions": positions,
    }
    ctx = build_cluster_track_context(
        query,
        road_graph=road_graph,
        camera_to_node=camera_to_node,
        shortest_paths={},
        planning_protocol="road_intents_v1",
        trajectory_scope="single_cluster",
        road_intent_config={"max_waypoints": 2},
    )
    return build_camera_mapping_context(ctx, road_graph=road_graph)


def _extract_experience_extra_nodes(active_experience):
    """Collect retrieved candidate/teacher road nodes W_q for corridor subgraph construction."""
    if not isinstance(active_experience, dict):
        return None
    nodes = set()
    for row in (active_experience.get("teacher") or {}).get("mapping") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            nodes.add(int(row[1]))
    for row in active_experience.get("camera_support") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            for item in row[2] or []:
                if isinstance(item, (list, tuple)) and len(item) >= 3 and (int(item[1]) + int(item[2])) > 0:
                    nodes.add(int(item[0]))
    return sorted(nodes) or None


def _make_llm_road_intent_callable(camera_to_node, cam_xy_dict):
    """Build a live C2 road-intent callable backed by LLMClient on corridor subgraph G_q."""
    def _call(*, camera_trace, corridor_subgraph, model, retrieved_experience=None, **_kwargs):
        ctx = _build_public_mapping_context(
            camera_trace,
            road_graph=corridor_subgraph,
            camera_to_node=camera_to_node,
            cam_xy_dict=cam_xy_dict,
        )
        if ctx is None:
            return {"camera_node_choices": {}, "interval_intents": {}}
        prompt = build_camera_mapping_prompt(
            ctx,
            include_pair_evidence=True,
            road_graph=corridor_subgraph,
            retrieved_experience=retrieved_experience,
        )
        try:
            raw_text = LLMClient.call_llm(prompt, model=model)
            groups, _, mapping = parse_camera_mapping_response(raw_text, ctx, road_graph=corridor_subgraph)
        except Exception:
            return {"camera_node_choices": {}, "interval_intents": {}, "fallback_on_error": True}
        obs_id_to_idx = {
            obs["obs_id"]: int(obs["input_index"])
            for obs in ctx.get("observations", [])
            if "obs_id" in obs and "input_index" in obs
        }
        parsed_tracks = []
        for trk in groups.get("tracks") or []:
            t_indices = [
                obs_id_to_idx[oid]
                for oid in (trk.get("obs_ids") or [])
                if oid in obs_id_to_idx
            ]
            t_intents = {}
            t_pair_intents = {}
            for seg_idx, item in enumerate(trk.get("interval_intents") or []):
                via_list = [int(n) for n in (item.get("via_nodes") or [])]
                t_intents[seg_idx] = via_list
                f_oid, t_oid = item.get("from_obs_id"), item.get("to_obs_id")
                if f_oid in obs_id_to_idx and t_oid in obs_id_to_idx:
                    t_pair_intents[(obs_id_to_idx[f_oid], obs_id_to_idx[t_oid])] = via_list
            if t_indices:
                parsed_tracks.append({
                    "retained_obs_indices": t_indices,
                    "interval_intents": t_intents,
                    "interval_intents_by_obs_pair": t_pair_intents,
                })
        primary = parsed_tracks[0] if parsed_tracks else {}
        return {
            "camera_node_choices": mapping,
            "interval_intents": primary.get("interval_intents", {}),
            "interval_intents_by_obs_pair": primary.get("interval_intents_by_obs_pair", {}),
            "retained_obs_indices": primary.get("retained_obs_indices"),
            "parsed_tracks": parsed_tracks,
        }

    return _call


def load_road_assets(args):
    """Load prepared road-network graph, camera bindings, shortest paths, and ground truth."""
    with open(args.road_graph_file, "rb") as f:
        road_graph = pickle.load(f)
    with open(args.cam_xy_dict_file, "rb") as f:
        cam_xy_dict = pickle.load(f)
    with open(args.cid_rid_correspondence_file, "rb") as f:
        cid_to_rid = pickle.load(f)
    camera_to_node = (
        {int(k): int(v) for k, v in cid_to_rid.items()}
        if isinstance(cid_to_rid, dict)
        else {int(item["id"]): int(item["node_id"]) for item in cid_to_rid}
    )
    with open(args.rid_shortest_path_results_file, "rb") as f:
        shortest_paths = pickle.load(f)
    with open(args.road_node_traj_gt_file, "rb") as f:
        traj_gt_dict = pickle.load(f)
    return road_graph, camera_to_node, cam_xy_dict, shortest_paths, traj_gt_dict


def recover_cluster_trajectory(
    camera_trace,
    *,
    road_graph,
    camera_to_node,
    shortest_paths,
    cam_xy_dict=None,
    router: PerformanceAwareMultiLLMRouter | None = None,
    llm_callable=None,
    experience_bank: Dict[str, Any] | None = None,
    query_source_group: str | None = None,
    retrieved_experience: Any | None = None,
    retrieval_similarity: float | None = None,
    gate_accepted: bool = False,
    interval_intents: Dict[int, list] | None = None,
    sp_cache: Dict[Any, Any] | None = None,
    max_speed_mps: float = 25.0,
    max_span_m: float = 1200.0,
    min_dwell_sec: float = 30.0,
    detour_ratio_threshold: float = 1.05,
    detour_excess_m: float = 0.0,
    enable_dp_corridor_separation: bool = False,
    max_intersection_cam_id: int | None = None,
) -> Dict[str, Any]:
    """Execute the closed-loop C1 -> C3 -> C2 trajectory recovery pipeline (Algorithm 1)."""
    active_mapping = dict(camera_to_node or {})
    dwell_adj = max(0.0, min(float(min_dwell_sec), 30.0))
    dp_corridors = None
    if enable_dp_corridor_separation:
        dp_corridors = extract_kinematic_dp_corridors(
            camera_trace,
            camera_to_node=active_mapping,
            shortest_paths=shortest_paths,
            road_graph=road_graph,
            max_intersection_cam_id=max_intersection_cam_id,
        )
    if dp_corridors:
        merged_cam_trace = list(dp_corridors[0]["camera_trace"])
        rid_trace = list(dp_corridors[0]["rid_trace"])
    else:
        merged_cam_trace = sorted(merge_tm_adj_points(list(camera_trace), adj_range=dwell_adj), key=lambda x: x[1])
        rid_trace = [
            (int(active_mapping[cam]), float(ts))
            for cam, ts in merged_cam_trace
            if cam in active_mapping
        ]
        rid_trace = sorted(merge_tm_adj_points(rid_trace, adj_range=dwell_adj), key=lambda x: x[1])

    # Stage 1 (C1): Evaluate experience bank B(O_q) (if supplied) and invocation gate gamma_inv(O_q)
    retrieval_report = None
    active_experience = retrieved_experience
    active_gate_accepted = bool(gate_accepted) or (active_experience is not None)
    if isinstance(experience_bank, dict) and "strict_index" in experience_bank and "local_index" in experience_bank:
        mapping_ctx = _build_public_mapping_context(
            merged_cam_trace,
            road_graph=road_graph,
            camera_to_node=active_mapping,
            cam_xy_dict=cam_xy_dict,
        )
        if mapping_ctx is not None:
            local_cases = experience_bank["local_index"].get("cases", {})
            retrieval_report = retrieve_mapping_experience_with_local_fallback(
                mapping_ctx,
                experience_bank["strict_index"],
                experience_bank["local_index"],
                source_eligible_ids=experience_bank.get("source_eligible_ids", list(local_cases.keys())),
                candidate_rows=experience_bank.get("candidate_rows", []),
                local_gate=experience_bank.get("local_gate", "g5_kinematic_boundary_v1"),
                road_graph=road_graph,
                teacher_kinematics_index=experience_bank.get("teacher_kinematics_index", {}),
                v_stop_mps=float(experience_bank.get("v_stop_mps", 5.0)),
                exclude_source_group=query_source_group,
            )
            if retrieval_report.get("gate", {}).get("inject") is True:
                active_gate_accepted = True
                active_experience = retrieval_report.get("payload")

    bank_candidate_bindings = _extract_bank_candidate_bindings(active_mapping, active_experience)
    gate_info = evaluate_cluster_invocation_gate(
        merged_cam_trace,
        road_graph=road_graph,
        camera_to_node=active_mapping,
        shortest_paths=shortest_paths,
        max_speed_mps=float(max_speed_mps),
        max_span_m=float(max_span_m),
        min_dwell_sec=float(min_dwell_sec),
        detour_ratio_threshold=float(detour_ratio_threshold),
        detour_excess_m=float(detour_excess_m),
        candidate_bindings=bank_candidate_bindings,
    )
    if not gate_info["invoke_llm"]:
        verified = execute_three_tier_corridor_verification(
            rid_trace,
            road_graph=road_graph,
            shortest_paths=shortest_paths,
            max_speed_mps=float(max_speed_mps),
            max_span_m=float(max_span_m),
            min_dwell_sec=float(min_dwell_sec),
            detour_ratio_threshold=float(detour_ratio_threshold),
            detour_excess_m=float(detour_excess_m),
        )
        secondary_tracks = []
        for sub_c in (dp_corridors or [])[1:]:
            sub_v = execute_three_tier_corridor_verification(
                list(sub_c["rid_trace"]),
                road_graph=road_graph,
                shortest_paths=shortest_paths,
                max_speed_mps=float(max_speed_mps),
                max_span_m=float(max_span_m),
                min_dwell_sec=float(min_dwell_sec),
                detour_ratio_threshold=float(detour_ratio_threshold),
                detour_excess_m=float(detour_excess_m),
            )
            if sub_v["trajectory"]:
                secondary_tracks.append(sub_v["trajectory"])
        return {
            "trajectory": verified["trajectory"],
            "tracks": [verified["trajectory"], *secondary_tracks] if verified["trajectory"] else secondary_tracks,
            "invoked_llm": False,
            "gate_info": gate_info,
            "retrieval_report": retrieval_report,
            "subgraph_stats": None,
            "routing_decision": None,
            "verification_audit": verified,
        }

    # Stage 2 (C3): Corridor subgraph pruning (G -> G_q) and difficulty-aware routing
    camera_candidates = _build_top2_camera_candidates(
        road_graph, cam_xy_dict, (cam for cam, _ in merged_cam_trace)
    )
    extra_nodes = _extract_experience_extra_nodes(active_experience)
    corridor_subgraph, subgraph_stats = build_cluster_corridor_subgraph(
        merged_cam_trace,
        road_graph=road_graph,
        camera_to_node=active_mapping,
        camera_candidates=camera_candidates,
        shortest_paths=shortest_paths,
        sp_cache=sp_cache if sp_cache is not None else {},
        pair_scope="consecutive_and_two_hop",
        k_hop=1,
        extra_nodes=extra_nodes,
    )
    if retrieval_similarity is not None:
        ret_sim = float(retrieval_similarity)
    elif active_gate_accepted:
        audit_dict = (retrieval_report or {}).get("audit") or {}
        cov = audit_dict.get("complete_current_pair_coverage")
        denom = max(1, len(merged_cam_trace) - 1)
        if isinstance(cov, (int, float)):
            ret_sim = float(cov) / denom if float(cov) > 1.0 else float(cov)
        else:
            ret_sim = 1.0
    else:
        ret_sim = 0.0
    prompt_char_len = int(3200 + 310 * subgraph_stats["corridor_node_count"] + (2200 if active_gate_accepted else 0))
    if isinstance(cam_xy_dict, dict) and corridor_subgraph.number_of_nodes() >= 2:
        try:
            prompt_ctx = _build_public_mapping_context(
                merged_cam_trace,
                road_graph=corridor_subgraph,
                camera_to_node=active_mapping,
                cam_xy_dict=cam_xy_dict,
            )
            if prompt_ctx is not None:
                prompt_char_len = len(
                    build_camera_mapping_prompt(
                        prompt_ctx,
                        include_pair_evidence=True,
                        road_graph=corridor_subgraph,
                        retrieved_experience=active_experience,
                    )
                )
        except Exception:
            pass
    case_features = build_case_features_from_trace(
        merged_cam_trace,
        cam_xy_dict=cam_xy_dict,
        rid_trace=rid_trace,
        input_char_length=prompt_char_len,
        retrieval_count=1 if active_gate_accepted else 0,
        retrieval_similarity=ret_sim,
        road_graph=road_graph,
        camera_to_node=active_mapping,
        time_unit="second",
        coordinate_unit="meter",
    )
    routing_decision = None
    selected_model = None
    if router is not None:
        routing_decision = router.route(
            case_features,
            gate_aware=True,
            gate_accepted=bool(active_gate_accepted),
        )
        selected_model = routing_decision.get("selected_model")

    # Stage 3 (C2): Road-intent generation and three-tier verification on G_q
    planned_intents = dict(interval_intents or {})
    if callable(llm_callable):
        llm_output = llm_callable(
            camera_trace=merged_cam_trace,
            corridor_subgraph=corridor_subgraph,
            model=selected_model,
            case_features=case_features,
            retrieved_experience=active_experience,
        )
        if isinstance(llm_output, dict):
            node_choices = llm_output.get("camera_node_choices")
            if isinstance(node_choices, dict) and node_choices:
                active_mapping = {**active_mapping, **{int(k): int(v) for k, v in node_choices.items()}}
            retained_indices = llm_output.get("retained_obs_indices")
            obs_pair_intents = llm_output.get("interval_intents_by_obs_pair")
            if isinstance(retained_indices, list):
                indexed_obs = [
                    (orig_idx, int(active_mapping[merged_cam_trace[orig_idx][0]]), float(merged_cam_trace[orig_idx][1]))
                    for orig_idx in retained_indices
                    if 0 <= orig_idx < len(merged_cam_trace) and merged_cam_trace[orig_idx][0] in active_mapping
                ]
                cleaned_anchors = []
                anchor_orig_sets = []
                for orig_idx, node, ts in indexed_obs:
                    if not cleaned_anchors:
                        cleaned_anchors.append((node, ts))
                        anchor_orig_sets.append({orig_idx})
                        continue
                    prev_node, prev_ts = cleaned_anchors[-1]
                    if ts < prev_ts:
                        continue
                    if node == prev_node and (ts - prev_ts) <= float(min_dwell_sec):
                        anchor_orig_sets[-1].add(orig_idx)
                        continue
                    cleaned_anchors.append((node, ts))
                    anchor_orig_sets.append({orig_idx})
                rid_trace = cleaned_anchors
                if isinstance(obs_pair_intents, dict) and obs_pair_intents:
                    planned_intents = {}
                    for seg_idx in range(len(cleaned_anchors) - 1):
                        left_set, right_set = anchor_orig_sets[seg_idx], anchor_orig_sets[seg_idx + 1]
                        for (f_idx, t_idx), via_list in obs_pair_intents.items():
                            if f_idx in left_set and t_idx in right_set:
                                planned_intents[seg_idx] = list(via_list)
                                break
                elif isinstance(llm_output.get("interval_intents"), dict):
                    planned_intents = {int(k): list(v) for k, v in llm_output["interval_intents"].items()}
            else:
                if isinstance(node_choices, dict) and node_choices:
                    rid_trace = [
                        (int(active_mapping[cam]), float(ts))
                        for cam, ts in merged_cam_trace
                        if cam in active_mapping
                    ]
                    rid_trace = sorted(merge_tm_adj_points(rid_trace, adj_range=dwell_adj), key=lambda x: x[1])
                if isinstance(llm_output.get("interval_intents"), dict):
                    planned_intents = {int(k): list(v) for k, v in llm_output["interval_intents"].items()}

    exec_graph = corridor_subgraph if corridor_subgraph.number_of_nodes() > 0 else road_graph
    verified = execute_three_tier_corridor_verification(
        rid_trace,
        interval_intents=planned_intents,
        road_graph=exec_graph,
        shortest_paths=shortest_paths,
        max_speed_mps=float(max_speed_mps),
        max_span_m=float(max_span_m),
        min_dwell_sec=float(min_dwell_sec),
        detour_ratio_threshold=float(detour_ratio_threshold),
        detour_excess_m=float(detour_excess_m),
    )
    verified_tracks = list(verified.get("tracks") or ([verified["trajectory"]] if verified["trajectory"] else []))
    for sub_c in (dp_corridors or [])[1:]:
        sub_v = execute_three_tier_corridor_verification(
            list(sub_c["rid_trace"]),
            road_graph=exec_graph,
            shortest_paths=shortest_paths,
            max_speed_mps=float(max_speed_mps),
            max_span_m=float(max_span_m),
            min_dwell_sec=float(min_dwell_sec),
            detour_ratio_threshold=float(detour_ratio_threshold),
            detour_excess_m=float(detour_excess_m),
        )
        if sub_v["trajectory"]:
            verified_tracks.append(sub_v["trajectory"])
    return {
        "trajectory": verified["trajectory"],
        "tracks": verified_tracks,
        "invoked_llm": bool(callable(llm_callable)),
        "gate_info": gate_info,
        "retrieval_report": retrieval_report,
        "subgraph_stats": subgraph_stats,
        "routing_decision": routing_decision,
        "verification_audit": verified,
    }


def run_pipeline(args) -> Dict[str, Any]:
    """Execute trajectory recovery and multi-metric evaluation."""
    dataset = str(getattr(args, "dataset", "uv"))
    records_all = load_car_feat(args.record_file, args.fps, getattr(args, "use_plate", True))
    if dataset in {"uv", "uv_z"}:
        for record in records_all:
            record["time"] += 10000

    if getattr(args, "sort_record", True):
        sorted_indices = np.argsort([r["time"] for r in records_all], kind="stable")
        records = [records_all[i] for i in sorted_indices]
    else:
        records = records_all

    clustering_res_file = getattr(args, "clustering_res_file", "")
    if clustering_res_file and os.path.exists(clustering_res_file):
        with open(clustering_res_file, "rb") as f:
            labels = pickle.load(f)
    else:
        from cluster_algorithm import SigCluster

        sig_cfg = getattr(args, "sig_cluster", {}) or {}
        feat_dim = int(getattr(args, "feat_dim", 256))
        f_car = [x["car_feature"] for x in records]
        f_plate = [x["plate_feature"] for x in records]
        sig_cluster = SigCluster(
            feature_dims=[feat_dim, feat_dim, feat_dim],
            ngpu=int(getattr(args, "ngpu", 1)),
            useFloat16=bool(sig_cfg.get("useFloat16", False)),
        )
        labels = sig_cluster.fit(
            [[a, b, a] for a, b in zip(f_car, f_plate)],
            None,
            weights=sig_cfg.get("weights", [0.5, 0.5, 0.0]),
            similarity_threshold=float(sig_cfg.get("sim_thres", 0.5)),
            topK=int(sig_cfg.get("topK", 50)),
            query_num=int(sig_cfg.get("query_num", 1000)),
            normalization=True,
        )

    clustering_metrics = evaluate(records, labels, log=True)
    road_graph, camera_to_node, cam_xy_dict, shortest_paths, traj_gt_dict = load_road_assets(args)
    processed_paths = preprocess_paths_with_length(shortest_paths, road_graph)
    vid2cid = gen_vid2cid(records, labels)

    cluster_cam_traces = defaultdict(list)
    for rec, cid in zip(records, labels):
        if int(cid) >= 0:
            cluster_cam_traces[int(cid)].append((int(rec["camera_id"]), float(rec["time"])))

    routing_cfg = getattr(args, "llm_routing", {}) or {}
    candidate_models = routing_cfg.get("candidate_models") or ["gpt-4.1-nano", "gpt-4.1", "qwen3.5-plus"]
    default_model = routing_cfg.get("default_model", candidate_models[0])
    quality_tolerance = float(routing_cfg.get("quality_tolerance", 0.01))
    router = PerformanceAwareMultiLLMRouter(
        candidate_models,
        default_model=default_model,
        delta=quality_tolerance,
        seed=int(getattr(args, "seed", 0)),
    )
    ckpt_file = routing_cfg.get("router_ckpt_file")
    if ckpt_file and os.path.exists(ckpt_file):
        router, _ = PerformanceAwareMultiLLMRouter.load(ckpt_file)

    experience_bank = None
    bank_file = getattr(args, "experience_bank_file", "")
    if bank_file and os.path.exists(bank_file):
        with open(bank_file, "rb") as f:
            experience_bank = pickle.load(f)

    llm_callable = None
    if getattr(args, "allow_api", False):
        LLMClient.configure(getattr(args, "llm_client", {}) or {})
        llm_callable = _make_llm_road_intent_callable(camera_to_node, cam_xy_dict)

    recovered_traj_dict: Dict[int, list] = {}
    sp_cache: Dict[Any, Any] = {}
    max_speed = float(getattr(args, "max_speed_mps", 25.0))
    max_span = float(getattr(args, "max_span_m", 1200.0))
    min_dwell = float(getattr(args, "min_dwell_sec", 30.0))
    detour_ratio = float(getattr(args, "detour_ratio_threshold", 1.05))
    detour_excess = float(getattr(args, "detour_excess_m", 0.0))
    enable_dp_sep = bool(getattr(args, "enable_dp_corridor_separation", False))
    max_inter_cam = getattr(args, "max_intersection_cam_id", None)
    max_inter_cam = None if max_inter_cam is None else int(max_inter_cam)

    for cid, raw_trace in sorted(cluster_cam_traces.items()):
        outcome = recover_cluster_trajectory(
            raw_trace,
            road_graph=road_graph,
            camera_to_node=camera_to_node,
            shortest_paths=processed_paths,
            cam_xy_dict=cam_xy_dict,
            router=router,
            llm_callable=llm_callable,
            experience_bank=experience_bank,
            sp_cache=sp_cache,
            max_speed_mps=max_speed,
            max_span_m=max_span,
            min_dwell_sec=min_dwell,
            detour_ratio_threshold=detour_ratio,
            detour_excess_m=detour_excess,
            enable_dp_corridor_separation=enable_dp_sep,
            max_intersection_cam_id=max_inter_cam,
        )
        recovered_traj_dict[cid] = outcome["trajectory"]

    scale_coef_list = [
        float(x) for x in getattr(args, "scale_coef", [0.04267, 0.04267, 0.04267])
    ]
    eval_summary = eval_trajs(
        records,
        labels,
        traj_gt_dict,
        recovered_traj_dict,
        road_graph,
        scale_coef_list,
        vid2cid,
        "recovered",
        True,
    )
    return {
        "clustering": clustering_metrics,
        "trajectory_recovery": eval_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Road-Network Trajectory Recovery Pipeline")
    parser.add_argument("--cfg_file", type=str, default="config/uv.yaml", help="Path to YAML configuration file")
    parser.add_argument("--allow-api", dest="allow_api", action="store_true", help="Enable live LLM API requests")
    known_args, _ = parser.parse_known_args()
    cfg = yaml_config_hook(known_args.cfg_file)
    add_config_arguments(parser, flatten_dict(cfg))
    args = parser.parse_args()
    for key, value in unflatten_dict(vars(args)).items():
        setattr(args, key, value)
    args.allow_api = known_args.allow_api
    results = run_pipeline(args)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
