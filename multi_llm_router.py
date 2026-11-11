"""Performance-Aware Multi-LLM Router (PerformanceAwareMultiLLMRouter): a supervised API router trained exclusively on measured recovery records.

Architectural boundaries:
- Trains two independent GradientBoostingRegressor models per candidate API to predict recovery quality (quality_f1) and recovery latency (recovery_latency_sec).
- Uses only the 9 pre-invocation observable features (FEATURE_FIELDS): applies log1p to non-negative quantities, clips ratios to [0, 1], and concatenates 9 missing-value indicators (missing values are distinct from measured zeros).
- Ground-truth trajectories, post-hoc difficulty, output token statistics, cost, and legacy IRT heuristics are strictly excluded from the predictors."""
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

# Compatibility aliases: legacy serialized checkpoints and scripts may import this module via
# performance_router or utils.performance_router.
sys.modules.setdefault("performance_router", sys.modules[__name__])
sys.modules.setdefault("utils.performance_router", sys.modules[__name__])

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "performance-router-v2-shared-observations"
LATENCY_TARGET = "successful_complete_recovery_v1"
FEATURE_FIELDS = (
    "traj_len", "time_span", "spatial_range", "branching_score", "missing_ratio",
    "jump_distance", "input_char_length", "retrieval_count", "retrieval_similarity",
)
_RATIO_FIELDS = {"branching_score", "missing_ratio", "retrieval_similarity"}
_SPLITS = ("train", "validation", "test")


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def latency_sample_eligible(row):
    """Only fully executed and evaluable recoveries provide valid completion latency; constraint violations are distinct from transport failures."""
    execution = row.get("execution_status")
    completed = (execution in {"completed", "success", "completed_with_fallback"} if execution is not None
                 else row.get("status") == "success")
    return (completed and row.get("status") in {"success", "invalid", "invalid_recovery", "fallback"}
            and row.get("quality_f1") is not None and row.get("has_usable_path") is not False
            and not row.get("cached"))


def primary_recovery_failed(row):
    """Retain primary recovery failure even when fallback succeeds, for enforcing routing failure-rate gates."""
    return row.get("primary_status", row.get("status")) != "success"


def build_case_feature_vector(case_features):
    """Return transformed numerical features in FEATURE_FIELDS order concatenated with missing-value indicators."""
    values, missing = [], []
    for field in FEATURE_FIELDS:
        value = _number(case_features.get(field))
        missing.append(float(value is None))
        values.append(0.0 if value is None else (
            float(np.clip(value, 0.0, 1.0)) if field in _RATIO_FIELDS
            else math.log1p(max(value, 0.0))))
    return np.asarray(values + missing, dtype=np.float64)


def build_case_features_from_trace(cam_trace, cam_xy_dict=None, rid_trace=None,
                                   input_char_length=None, retrieval_count=None,
                                   retrieval_similarity=None, *, road_graph=None,
                                   camera_to_node=None, time_unit="caller",
                                   coordinate_unit="unknown", coordinate_system=None):
    """Augment shared observation features with pre-invocation context statistics."""
    from utils.trajectory_features import build_observation_features
    base = build_observation_features(
        cam_trace, road_graph, camera_to_node, cam_xy_dict, rid_trace,
        time_unit=time_unit, coordinate_unit=coordinate_unit)
    base.update(input_char_length=_number(input_char_length),
                retrieval_count=_number(retrieval_count),
                retrieval_similarity=_number(retrieval_similarity))
    return {field: base.get(field) for field in FEATURE_FIELDS}


def validate_records(records):
    """Validate measured labels and retain only defined, non-secret fields."""
    rows, missing_groups = [], 0
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index}: expected a JSON object")
        missing = {"case_id", "api_id", "case_features", "quality_f1",
                   "recovery_latency_sec", "status"} - record.keys()
        if missing:
            raise ValueError(f"Record {index}: missing required fields {sorted(missing)}")
        if not isinstance(record["case_features"], dict):
            raise ValueError(f"Record {index}: case_features must be an object")
        for key in ("case_id", "api_id", "status"):
            if record[key] is None or not str(record[key]).strip():
                raise ValueError(f"Record {index}: {key} must be nonempty")
        quality, latency = _number(record["quality_f1"]), _number(record["recovery_latency_sec"])
        label = _number(record.get("router_quality_f1", record["quality_f1"]))
        observed_failure = (record.get("has_usable_path") is False
                            and record.get("failure_policy") == "no_usable_path_quality_zero"
                            and record["status"] in {"timeout", "error", "api_error", "parse_failure",
                                                     "technical_failure", "invalid", "invalid_recovery"})
        if ((quality is None and not (record["quality_f1"] is None and observed_failure and label == 0))
                or (quality is not None and not 0 <= quality <= 1)
                or label is None or not 0 <= label <= 1 or latency is None or latency < 0):
            raise ValueError(f"Record {index}: F1 must be in [0,1] and measured seconds >= 0")
        if quality is not None and label != quality:
            raise ValueError(f"Record {index}: router label must equal the original evaluable F1")
        if record.get("status") == "fallback":
            fallback_latency = _number(record.get("fallback_latency_sec"))
            fallback_policy = record.get("fallback_policy")
            if (record.get("fallback_used") is not True
                    or fallback_policy not in {"shortest_path", "rule_shortest_path"}
                    or record.get("fallback_status") != "success"
                    or record.get("execution_status") != "completed_with_fallback"
                    or record.get("has_usable_path") is not True
                    or quality is None or not record.get("primary_status")
                    or fallback_latency is None or fallback_latency < 0
                    or record.get("trajectory_source") != f"{fallback_policy}_fallback"):
                raise ValueError(f"Record {index}: successful fallback requires measured trajectory provenance")
            primary_latency = _number(record.get("primary_recovery_latency_sec"))
            if primary_latency is not None and latency + 1e-9 < primary_latency + fallback_latency:
                raise ValueError(f"Record {index}: recovery latency omits primary failure or fallback time")
        primary_quality = record.get("primary_quality_f1")
        if primary_quality is not None and (_number(primary_quality) is None or not 0 <= float(primary_quality) <= 1):
            raise ValueError(f"Record {index}: primary_quality_f1 must be an evaluable F1 or null")
        features = {}
        for field in FEATURE_FIELDS:
            value = record["case_features"].get(field)
            if value is not None and _number(value) is None:
                raise ValueError(f"Record {index}: invalid feature {field}")
            features[field] = _number(value)
        group_id = record.get("group_id")
        group_missing = bool(record.get("group_id_missing")) or group_id is None or not str(group_id).strip()
        if group_missing:
            missing_groups += 1
        if group_id is None or not str(group_id).strip():
            group_id = record["case_id"]
        row = {"case_id": str(record["case_id"]), "group_id": str(group_id),
               "api_id": str(record["api_id"]), "case_features": features,
               "quality_f1": quality, "router_quality_f1": label, "recovery_latency_sec": latency,
               "status": str(record["status"]), "group_id_missing": group_missing}
        if "split" in record:
            split = {"val": "validation", "valid": "validation"}.get(record["split"], record["split"])
            if split not in _SPLITS:
                raise ValueError(f"Record {index}: split must be train/validation/test")
            row["split"] = split
        for field in ("feature_extraction_sec", "routing_overhead_sec", "end_to_end_latency_sec",
                      "online_preparation_sec", "fallback_latency_sec", "primary_recovery_latency_sec"):
            value = record.get(field)
            if field == "feature_extraction_sec" and value is None:
                value = record.get("feature_extraction_latency_sec")
            if value is not None:
                value = _number(value)
                if value is None or value < 0:
                    raise ValueError(f"Record {index}: invalid timing {field}")
                row[field] = value
        for field in ("data_mode", "synthetic", "data_kind", "run_id", "repeat_id", "measurement_id",
                      "pipeline_version", "prompt_version", "config_hash", "api_config_hash",
                      "feature_version", "recovery_context_hash", "retrieval_assets_hash", "label_source", "cached", "cache_source_id",
                      "has_usable_path", "execution_status", "recovery_status", "failure_type",
                      "failure_policy", "latency_source", "trajectory_source", "timestamp",
                      "started_at", "finished_at", "started_at_utc", "finished_at_utc", "model", "response_model", "endpoint_sha256",
                      "prompt_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
                      "api_requests", "refinement_rounds", "fallback_used", "fallback_policy", "fallback_trigger",
                      "fallback_status", "primary_status", "primary_execution_status", "primary_quality_f1",
                      "fallback_naive_rules", "fallback_baseline_diagnostics"):
            if field in record:
                row[field] = record[field]
        rows.append(row)
    if not rows:
        raise ValueError("No measured recovery records; synthetic fallback is disabled")
    if missing_groups:
        logger.warning("%d records lack group_id; using case_id cannot prevent related-case leakage", missing_groups)
    # Each case corresponds to a distinct pre-invocation input; repeated API calls for the same case must share identical features.
    signatures = {}
    for row in rows:
        signature = tuple(build_case_feature_vector(row["case_features"]))
        previous = signatures.setdefault(row["case_id"], signature)
        if previous != signature:
            raise ValueError(f"case_id {row['case_id']!r} has inconsistent input features")
    return rows






class PerformanceAwareMultiLLMRouter:
    """Fits quality and latency predictors offline and selects APIs online, returning explicit fallback reasons on anomalies."""

    def __init__(self, model_names, default_model=None, delta=.01, seed=0, predictor_params=None):
        self.model_names = list(model_names)
        if not self.model_names or len(set(self.model_names)) != len(self.model_names):
            raise ValueError("Candidate API ids must be nonempty and unique")
        if any(not isinstance(name, str) or not name.strip() for name in self.model_names):
            raise ValueError("Candidate API ids must be nonempty strings")
        self.default_model = default_model or self.model_names[0]
        if _number(delta) is None or delta < 0 or delta > 1:
            raise ValueError("delta must be an absolute F1 difference in [0,1]")
        self.delta, self.seed = float(delta), int(seed)
        self.predictor_params = {"n_estimators": 120, "max_depth": 3,
                                 "min_samples_leaf": 1, "learning_rate": .08}
        self.predictor_params.update(predictor_params or {})
        self.predictor_params["random_state"] = self.seed
        self.quality_predictor = self.latency_predictor = None
        self.training_data_mode = "unknown"
        self.routing_strategy = "learned"
        self.fixed_api = None
        self.fallback_reason = None
        self.training_profiles = {}

    def _matrix(self, features, apis):
        one_hot = np.eye(len(self.model_names))
        return np.asarray([np.concatenate((build_case_feature_vector(case),
                           one_hot[self.model_names.index(api)]))
                           for case, api in zip(features, apis)])

    def fit(self, records):
        """Validate measured records and fit both regressors; the caller is responsible for train/validation/test group isolation."""
        rows = validate_records([row for row in records if not row.get("cached")])
        if any(row.get("split", "train") != "train" for row in rows):
            raise ValueError("Predictor fitting accepts only the training split")
        modes = {"mock" if row.get("data_mode") == "mock" or row.get("synthetic")
                 or row.get("data_kind") == "synthetic_smoke_only" else "real" for row in rows}
        if len(modes) != 1:
            raise ValueError("Mock and measured training labels cannot be mixed")
        self.training_data_mode = modes.pop()
        unknown = {row["api_id"] for row in rows} - set(self.model_names)
        if unknown:
            raise ValueError(f"Training records contain unconfigured API ids: {sorted(unknown)}")
        for api in self.model_names:
            cases = {row["case_id"] for row in rows if row["api_id"] == api}
            if len(cases) < 2:
                raise ValueError(f"API {api!r} needs measured training data for at least 2 distinct cases")
        latency_rows = [row for row in rows if latency_sample_eligible(row)]
        for api in self.model_names:
            if not any(row["api_id"] == api for row in latency_rows):
                raise ValueError(f"API {api!r} has no successful complete recovery latency samples")
        matrix = self._matrix([row["case_features"] for row in rows], [row["api_id"] for row in rows])
        self.quality_predictor = GradientBoostingRegressor(**self.predictor_params)
        self.latency_predictor = GradientBoostingRegressor(**self.predictor_params)
        self.quality_predictor.fit(matrix, [row["router_quality_f1"] for row in rows])
        latency_matrix = self._matrix([row["case_features"] for row in latency_rows],
                                      [row["api_id"] for row in latency_rows])
        self.latency_predictor.fit(latency_matrix, [row["recovery_latency_sec"] for row in latency_rows])
        self.training_profiles = {}
        for api in self.model_names:
            observed = [row for row in rows if row["api_id"] == api]
            durations = [row["recovery_latency_sec"] for row in latency_rows if row["api_id"] == api]
            starts = [row.get("started_at") or row.get("started_at_utc") or row.get("timestamp") for row in observed]
            finishes = [row.get("finished_at") or row.get("finished_at_utc") or row.get("timestamp") for row in observed]
            self.training_profiles[api] = {
                "records": len(observed), "cases": len({row["case_id"] for row in observed}),
                "successful_records": len(durations),
                "failure_rate": sum(primary_recovery_failed(row) for row in observed) / len(observed),
                "fallback_calls": sum(row.get("fallback_used") is True for row in observed),
                "fallback_rate": sum(row.get("fallback_used") is True for row in observed) / len(observed),
                "completion_failure_rate": 1 - len(durations) / len(observed),
                "mean_router_quality_f1": float(np.mean([row["router_quality_f1"] for row in observed])),
                "successful_recovery_latency_sec": {"count": len(durations), "mean": float(np.mean(durations)),
                    "median": float(np.median(durations)), "p95": float(np.percentile(durations, 95)),
                    "p95_unstable": len(durations) < 20},
                "run_ids": sorted({row["run_id"] for row in observed if row.get("run_id")}),
                "api_config_hashes": sorted({row["api_config_hash"] for row in observed if row.get("api_config_hash")}),
                "collection_started_at": min(filter(None, starts), default=None),
                "collection_finished_at": max(filter(None, finishes), default=None),
            }
        self.routing_strategy, self.fixed_api, self.fallback_reason = "learned", None, None
        return self

    def _predict(self, features, apis):
        matrix = self._matrix([features] * len(apis), apis)
        quality = np.asarray(self.quality_predictor.predict(matrix))
        latency = np.asarray(self.latency_predictor.predict(matrix))
        if not np.isfinite(quality).all() or not np.isfinite(latency).all():
            raise ValueError("Predictors returned nonfinite values")
        return {api: {"predicted_accuracy": float(np.clip(q, 0, 1)),
                      "predicted_latency_sec": max(float(t), 1e-6)}
                for api, q, t in zip(apis, quality, latency)}

    def route(
        self,
        case_features,
        candidate_models=None,
        *,
        gate_aware=False,
        gate_accepted=None,
        fast_model=None,
        **legacy_ignored,
    ):
        """Filter candidate APIs within delta of the best predicted F1, then select the one with the lowest predicted latency.

        When gate_aware=True (Section III-C, Eq. 3), clusters whose C1 retrieval gate abstains
        (gate_accepted=False or retrieval_count == 0) are dispatched directly to fast_model
        (or default_model), while gate-accepted clusters are routed via the GBR Pareto selector.
        """
        requested = self.model_names if candidate_models is None else list(candidate_models)
        reason, scores = None, {}
        selected = self.default_model
        strategy_label = "learned"
        if not requested:
            reason = "no_available_candidates"
        elif set(requested) - set(self.model_names):
            reason = "unknown_candidate_api"
        elif self.quality_predictor is None or self.latency_predictor is None:
            reason = self.fallback_reason or "model_not_loaded"
        else:
            # Preserve the canonical API ordering from the checkpoint regardless of caller or set iteration order.
            apis = [api for api in self.model_names if api in requested]
            try:
                scores = self._predict(case_features, apis)
                if self.routing_strategy == "fixed_fallback":
                    if self.fixed_api not in apis:
                        reason = "fixed_fallback_api_unavailable"
                    else:
                        selected = self.fixed_api
                        reason = self.fallback_reason or "validation_quality_constraint_unsatisfied"
                elif gate_aware:
                    is_accepted = (
                        bool(gate_accepted)
                        if gate_accepted is not None
                        else (float(_number((case_features or {}).get("retrieval_count")) or 0.0) > 0.0)
                    )
                    if not is_accepted:
                        target_fast = fast_model or self.default_model
                        selected = target_fast if target_fast in apis else min(
                            apis, key=lambda api: (scores[api]["predicted_latency_sec"], self.model_names.index(api))
                        )
                        strategy_label = "gate_aware_fast"
                    else:
                        best = max(score["predicted_accuracy"] for score in scores.values())
                        eligible = [api for api in apis
                                    if scores[api]["predicted_accuracy"] + 1e-12 >= best - self.delta]
                        selected = min(eligible, key=lambda api: (
                            scores[api]["predicted_latency_sec"], -scores[api]["predicted_accuracy"],
                            self.model_names.index(api)))
                        strategy_label = "gate_aware_pareto"
                else:
                    best = max(score["predicted_accuracy"] for score in scores.values())
                    eligible = [api for api in apis
                                if scores[api]["predicted_accuracy"] + 1e-12 >= best - self.delta]
                    selected = min(eligible, key=lambda api: (
                        scores[api]["predicted_latency_sec"], -scores[api]["predicted_accuracy"],
                        self.model_names.index(api)))
            except Exception as exc:
                reason = f"prediction_error:{type(exc).__name__}"
                scores = {}
        prediction = scores.get(selected, {})
        return {"selected_api": selected, "selected_model": selected,
                "predicted_accuracy": prediction.get("predicted_accuracy"),
                "predicted_latency_sec": prediction.get("predicted_latency_sec"),
                "scores": scores, "delta": self.delta, "fallback_reason": reason,
                "routing_strategy": "fixed_fallback" if reason else strategy_label}

    def save(self, path, meta=None):
        # Checkpoints store only numerical model state and defined metadata.
        permitted = {"schema_version", "feature_fields", "feature_processing", "model_names",
                     "default_model", "data_source", "data_kind", "split_stats", "split_case_ids",
                     "split_group_ids", "predictor_params", "delta", "quality_tolerance",
                     "validation_metrics", "test_metrics", "reference_api", "routing_strategy",
                     "fallback_reason", "search_results", "seed", "timing_note", "measurement_context",
                     "api_config_hashes", "frozen_config_hash", "feature_version", "data_mode", "frozen",
                     "failure_policy", "training_profiles", "latency_target", "min_latency_reduction",
                     "require_no_failure_increase", "validation_gate"}
        metadata = {key: value for key, value in (meta or {}).items() if key in permitted}
        if self.training_data_mode == "mock":
            metadata.update(data_mode="mock", data_kind="synthetic_smoke_only")
        payload = {"training_data_mode": self.training_data_mode, "schema_version": SCHEMA_VERSION, "feature_fields": FEATURE_FIELDS,
                   "latency_target": LATENCY_TARGET, "training_profiles": self.training_profiles,
                   "model_names": self.model_names, "default_model": self.default_model,
                   "delta": self.delta, "seed": self.seed, "predictor_params": self.predictor_params,
                   "quality_predictor": self.quality_predictor, "latency_predictor": self.latency_predictor,
                   "routing_strategy": self.routing_strategy, "fixed_api": self.fixed_api,
                   "fallback_reason": self.fallback_reason, "metadata": metadata}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """Load and validate checkpoint schema, feature order, and model state without triggering retraining."""
        try:
            with Path(path).open("rb") as stream:
                payload = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as exc:
            raise ValueError("Unreadable/incompatible router checkpoint; retrain using measured recovery logs") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Incompatible/legacy router checkpoint; retrain using measured recovery logs")
        if payload.get("latency_target") != LATENCY_TARGET:
            raise ValueError("Incompatible latency target in checkpoint; retrain using successful complete recovery logs")
        try:
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("Checkpoint metadata must be an object")
            if tuple(payload.get("feature_fields", ())) != FEATURE_FIELDS:
                raise ValueError("Incompatible feature schema")
            router = cls(payload["model_names"], payload["default_model"], payload["delta"],
                         payload["seed"], payload["predictor_params"])
            router.training_data_mode = payload.get("training_data_mode", metadata.get("data_mode", "unknown"))
            router.training_profiles = payload.get("training_profiles", {})
            if router.training_data_mode == "mock":
                metadata.update(data_mode="mock", data_kind="synthetic_smoke_only")
            for field in ("quality_predictor", "latency_predictor", "routing_strategy", "fixed_api", "fallback_reason"):
                setattr(router, field, payload[field])
            for model in (router.quality_predictor, router.latency_predictor):
                check_is_fitted(model)
                if model.n_features_in_ != 2 * len(FEATURE_FIELDS) + len(router.model_names):
                    raise ValueError("Predictor input dimension does not match feature/API schema")
            if router.routing_strategy not in ("learned", "fixed_fallback"):
                raise ValueError("Invalid routing strategy")
            if router.routing_strategy == "fixed_fallback" and router.fixed_api not in router.model_names:
                raise ValueError("Unknown fixed fallback API")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid performance checkpoint; retrain: {exc}") from exc
        return router, metadata
