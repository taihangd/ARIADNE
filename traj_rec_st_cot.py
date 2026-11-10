import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

import requests

from utils.llm_request_options import (
    model_response_matches_request,
    normalize_request_options,
    normalize_response_model_names,
)
from utils.yaml_config_hook import resolve_api_key


# =============================================================================
# 1. LLM API Interface
# =============================================================================
class LLMClient:
    # Runtime defaults aligned with greedy single-call C2 execution; override via LLMClient.configure(cfg_dict).
    API_KEY: Optional[str] = None
    API_URL: Optional[str] = None
    PROVIDERS: Dict[str, Dict[str, Optional[str]]] = {}
    DEFAULT_MODEL: str = "gpt-4.1-nano"
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_SEED: Optional[int] = 0
    DEFAULT_MAX_TOKENS: int = 8192
    MAX_TOKENS_CAP: int = 8192
    MAX_RETRIES: int = 1
    LENGTH_RETRY_LIMIT: int = 0
    REQUEST_TIMEOUT: int = 180
    DEFAULT_REQUEST_OPTIONS: Dict = {}
    MODEL_REQUEST_OPTIONS: Dict[str, Dict] = {}
    DEFAULT_RESPONSE_MODEL_NAMES: list = []

    @staticmethod
    def _normalize_endpoint_url(raw_url_cfg, fallback: str = "https://api.openai.com/v1") -> str:
        raw_url = os.path.expandvars(str(raw_url_cfg)) if raw_url_cfg is not None else fallback
        if not raw_url or raw_url.startswith("$"):
            raw_url = fallback
        if not raw_url.rstrip("/").endswith("/chat/completions"):
            return raw_url.rstrip("/") + "/chat/completions"
        return raw_url

    @staticmethod
    def _try_resolve_key(sub_cfg: Dict) -> Optional[str]:
        try:
            return resolve_api_key(sub_cfg)
        except ValueError:
            return None

    @classmethod
    def configure(cls, cfg: Dict) -> None:
        """Inject LLM client parameters and per-provider official endpoints from configuration."""
        cfg = cfg or {}
        request_options = normalize_request_options(cfg.get("request_options"))
        response_model_names = normalize_response_model_names(cfg.get("response_model_names"))
        providers_cfg = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
        parsed_providers: Dict[str, Dict[str, Optional[str]]] = {}
        for name, p_cfg in providers_cfg.items():
            if isinstance(p_cfg, dict):
                parsed_providers[str(name).lower()] = {
                    "api_url": cls._normalize_endpoint_url(p_cfg.get("api_url")),
                    "api_key": cls._try_resolve_key(p_cfg),
                }
        cls.PROVIDERS = parsed_providers
        cls.API_KEY = cls._try_resolve_key(cfg) or next(
            (p["api_key"] for p in parsed_providers.values() if p.get("api_key")), None
        )
        if cls.API_KEY is None and not parsed_providers:
            cls.API_KEY = resolve_api_key(cfg)
        cls.API_URL = cls._normalize_endpoint_url(cfg.get("api_url", cls.API_URL or "https://api.openai.com/v1"))
        cls.DEFAULT_MODEL = str(cfg.get("default_model", "gpt-4.1-nano"))
        cls.DEFAULT_TEMPERATURE = float(cfg.get("default_temperature", 0.0))
        raw_seed = cfg.get("default_seed", 0)
        cls.DEFAULT_SEED = None if raw_seed is None else int(raw_seed)
        cls.DEFAULT_MAX_TOKENS = int(cfg.get("default_max_tokens", 8192))
        cls.MAX_TOKENS_CAP = int(cfg.get("max_tokens_cap", 8192))
        cls.MAX_RETRIES = int(cfg.get("max_retries", 1))
        cls.LENGTH_RETRY_LIMIT = int(cfg.get("length_retry_limit", 0))
        cls.REQUEST_TIMEOUT = int(cfg.get("request_timeout", 180))
        cls.DEFAULT_REQUEST_OPTIONS = request_options
        model_opts = cfg.get("model_request_options") if isinstance(cfg.get("model_request_options"), dict) else {}
        cls.MODEL_REQUEST_OPTIONS = {
            str(k).lower(): normalize_request_options(v)
            for k, v in model_opts.items()
            if isinstance(v, dict)
        }
        cls.DEFAULT_RESPONSE_MODEL_NAMES = response_model_names

    @classmethod
    def _resolve_provider_credentials(
        cls, model_name: str, api_url: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        if api_url is not None and api_key is not None:
            return api_url, api_key
        lower_model = str(model_name or "").lower()
        provider_key = "dashscope" if lower_model.startswith("qwen") else "openai"
        provider = cls.PROVIDERS.get(provider_key, {})
        resolved_url = api_url or provider.get("api_url") or cls.API_URL
        resolved_key = api_key or provider.get("api_key") or cls.API_KEY
        return resolved_url, resolved_key

    @classmethod
    def call_llm(
        cls,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        max_retries: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_tokens_cap: Optional[int] = None,
        length_retry_limit: Optional[int] = None,
        return_meta: bool = False,
        *,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        request_budget_hook=None,
        deadline: Optional[float] = None,
        request_options=None,
        response_model_names=None,
    ):
        """Call the LLM with automatic token expansion retry on length truncation.

        When optional parameters are not explicitly provided, class-level defaults injected by
        LLMClient.configure() are used; if configure() has not been called and API_KEY/API_URL
        are None, an error is raised immediately for easier debugging.
        request_options only adjusts whitelisted request fields; token limits remain governed by the call budget.
        """
        call_started = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        call_id = uuid4().hex
        model = cls.DEFAULT_MODEL if model is None else str(model)
        lower_model = model.lower()
        merged_req_opts = dict(cls.DEFAULT_REQUEST_OPTIONS)
        if lower_model.startswith("qwen"):
            merged_req_opts.setdefault("enable_thinking", False)
            merged_req_opts.setdefault("max_tokens_parameter", "max_completion_tokens")
        for prefix, m_opts in cls.MODEL_REQUEST_OPTIONS.items():
            if lower_model == prefix or lower_model.startswith(prefix + "-"):
                merged_req_opts.update(m_opts)
        if request_options is not None:
            merged_req_opts.update(normalize_request_options(request_options))
        request_options = normalize_request_options(merged_req_opts)
        response_model_names = normalize_response_model_names(
            cls.DEFAULT_RESPONSE_MODEL_NAMES if response_model_names is None else response_model_names)
        request_contract = {"request_options": dict(request_options),
                            "response_model_names": list(response_model_names)}
        api_url, api_key = cls._resolve_provider_credentials(model, api_url, api_key)
        if api_key is None or api_url is None:
            raise RuntimeError(
                "LLMClient API_KEY/API_URL not configured; call LLMClient.configure(cfg) first "
                "or provide api_key/api_url under the llm_client section in the YAML config"
            )
        temperature = cls.DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        seed = cls.DEFAULT_SEED if seed is None else int(seed)
        max_retries = cls.MAX_RETRIES if max_retries is None else int(max_retries)
        max_tokens = cls.DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
        max_tokens_cap = cls.MAX_TOKENS_CAP if max_tokens_cap is None else int(max_tokens_cap)
        length_retry_limit = cls.LENGTH_RETRY_LIMIT if length_retry_limit is None else int(length_retry_limit)
        request_timeout = cls.REQUEST_TIMEOUT if request_timeout is None else float(request_timeout)
        payload_options = {"temperature": temperature}
        if seed is not None:
            payload_options["seed"] = int(seed)
        if reasoning_effort is not None:
            payload_options["reasoning_effort"] = reasoning_effort
        token_parameter = request_options.pop("max_tokens_parameter", "max_tokens")
        for name, value in request_options.items():
            if value is None:
                payload_options.pop(name, None)
            else:
                payload_options[name] = value
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be a finite perf_counter timestamp")
        if not api_url.rstrip("/").endswith("/chat/completions"):
            api_url = api_url.rstrip("/") + "/chat/completions"

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        # Legacy max_retries specifies the maximum number of failed request attempts allowed
        # (at least one attempt). Retries triggered by length truncation use a separate budget
        # and are still counted in usage statistics even when the truncated response text is discarded.
        failed_attempts = 0
        length_retry_count = 0
        network_retry_count = 0
        current_max_tokens = int(max_tokens)
        max_tokens_cap = max(int(max_tokens_cap), int(max_tokens))
        max_attempts = max(1, max_retries)
        last_error, last_status = None, "error"
        attempts = []
        retry_after_failure = False
        retry_after_length = False

        def token_count(value):
            try:
                return int(value) if value is not None and int(value) >= 0 else None
            except (ValueError, TypeError, OverflowError):
                return None

        def accounting():
            def total(field):
                values = [attempt.get(field) for attempt in attempts]
                return sum(values) if all(value is not None for value in values) else None
            return {
                **request_contract,
                "call_id": call_id,
                "started_at_utc": started_at_utc,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "requested_model": model,
                "endpoint_sha256": hashlib.sha256(api_url.encode("utf-8")).hexdigest(),
                "reasoning_effort": payload_options.get("reasoning_effort"),
                # Current transport is non-streaming requests; no hidden SDK retries and no TTFT measured.
                "sdk_retry_count": 0,
                "ttft_sec": None,
                "transport_complete": attempts[-1]["transport_complete"] if attempts else False,
                "response_parse_success": attempts[-1]["response_parse_success"] if attempts else False,
                "error_category": attempts[-1].get("error_category") if attempts else None,
                "logical_call_count": 1,
                "api_request_count": len(attempts),
                "network_retry_count": network_retry_count,
                "length_retry_count": length_retry_count,
                "cached": False,
                "cache_hit": False,
                "api_attempts": [dict(attempt) for attempt in attempts],
                "discarded_attempts": [dict(attempt) for attempt in attempts
                                       if attempt.get("reason") != "accepted"],
                "billed_prompt_tokens": total("prompt_tokens"),
                "billed_completion_tokens": total("completion_tokens"),
                "billed_total_tokens": total("total_tokens"),
                "billed_reasoning_tokens": total("reasoning_tokens"),
                "billed_cached_prompt_tokens": total("cached_prompt_tokens"),
                "usage_complete": all(attempt.get("prompt_tokens") is not None
                                      and attempt.get("completion_tokens") is not None
                                      for attempt in attempts),
                "api_latency_sec": sum(attempt["request_latency_sec"] for attempt in attempts),
                "request_latency_sec": time.perf_counter() - call_started,
            }

        while failed_attempts < max_attempts:
            if deadline is not None and time.perf_counter() >= deadline:
                error = TimeoutError("LLM request deadline exceeded")
                error.status = "timeout"
                error.metadata = {**accounting(), "error_category": "TimeoutError"}
                raise error
            # The caller maintains a global persistent budget; reserve quota immediately before sending
            # each actual request, sharing the same budget across network and length-expansion retries.
            if request_budget_hook is not None:
                try:
                    request_budget_hook()
                except Exception as error:
                    error.budget_exhausted = True
                    error.status = "budget_exhausted"
                    error.metadata = {**accounting(), "budget_exhausted": True,
                                      "error_category": type(error).__name__}
                    error.api_attempts = error.metadata["api_attempts"]
                    error.discarded_attempts = error.metadata["discarded_attempts"]
                    raise
            # Budget reservation itself may take time; do not record an unsent request as an attempt if expired.
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                error = TimeoutError("LLM request deadline exceeded")
                error.status = "timeout"
                error.metadata = {**accounting(), "error_category": "TimeoutError"}
                raise error
            if retry_after_failure:
                network_retry_count += 1
                retry_after_failure = False
            if retry_after_length:
                length_retry_count += 1
                retry_after_length = False
            payload = {
                "model": model, "messages": messages, **payload_options,
                token_parameter: current_max_tokens,
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            attempt = {
                "attempt_id": len(attempts) + 1, "response_model": model,
                "call_id": call_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "prompt_tokens": None, "completion_tokens": None,
                "reasoning_tokens": None, "cached_prompt_tokens": None,
                "total_tokens": None, "max_tokens": current_max_tokens,
                "reason": "api_error", "request_latency_sec": 0.0,
                "transport_complete": False, "response_parse_success": False,
                "error_category": None,
            }
            attempts.append(attempt)
            req_start = time.perf_counter()
            try:
                timeout = request_timeout if remaining is None else min(request_timeout, remaining)
                try:
                    # stream=False consumes the full response body before returning rather than stopping at headers.
                    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout, stream=False)
                    attempt["transport_complete"] = True
                finally:
                    attempt["request_latency_sec"] = time.perf_counter() - req_start
                    attempt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                attempt["http_status"] = resp.status_code
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    pt = token_count(usage.get("prompt_tokens", usage.get("input_tokens")))
                    ct = token_count(usage.get("completion_tokens", usage.get("output_tokens")))
                    tt = token_count(usage.get("total_tokens"))
                    if tt is None and pt is not None and ct is not None:
                        tt = pt + ct
                    reasoning_tokens = token_count(
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
                    cached_prompt_tokens = token_count(
                        (usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", usage.get("prompt_cache_hit_tokens")))
                    response_model_reported = (isinstance(data.get("model"), str)
                                               and bool(data["model"].strip()))
                    response_model = str(data.get("model") or model)
                    attempt.update(response_model=response_model,
                                   response_model_reported=response_model_reported, prompt_tokens=pt,
                                   completion_tokens=ct, total_tokens=tt,
                                   reasoning_tokens=reasoning_tokens, cached_prompt_tokens=cached_prompt_tokens)
                    if not model_response_matches_request(model, response_model, response_model_names):
                        attempt["reason"] = "response_model_mismatch"
                        attempt["error_category"] = "ResponseModelMismatch"
                        last_error = ("response model mismatch: "
                                      f"requested={model}, response={response_model}")
                        last_status = "error"
                    else:
                        choice0 = data["choices"][0]
                        raw_content = choice0["message"]["content"]
                        finish_reason = str(choice0.get("finish_reason", ""))
                        attempt["response_parse_success"] = True
                        attempt["finish_reason"] = finish_reason
                        if (finish_reason.lower() == "length"
                                and length_retry_count < length_retry_limit
                                and current_max_tokens < max_tokens_cap):
                            attempt["reason"] = "length_retry"
                            current_max_tokens = min(
                                max_tokens_cap,
                                max(int(current_max_tokens * 1.8), current_max_tokens + 512),
                            )
                            retry_after_length = True
                            continue
                        text = _coerce_llm_content_to_text(raw_content)
                        attempt["reason"] = "accepted"
                        if return_meta:
                            return {
                                "text": text, "finish_reason": finish_reason,
                                "max_tokens": current_max_tokens, "usage": usage,
                                "prompt_tokens": pt, "completion_tokens": ct,
                                "total_tokens": tt, "response_model": response_model,
                                "reasoning_tokens": reasoning_tokens, "cached_prompt_tokens": cached_prompt_tokens,
                                "response_model_reported": response_model_reported,
                                **accounting(),
                            }
                        return text
                else:
                    last_error, last_status = f"HTTP {resp.status_code}", "error"
                    attempt["reason"] = "http_error"
                    attempt["error_category"] = "HTTPError"
            except Exception as error:
                # Exception messages may contain credentials, so record only the exception type name.
                last_error = type(error).__name__
                last_status = ("timeout" if isinstance(error, (requests.Timeout, TimeoutError))
                               else "parse_failure" if isinstance(error, (ValueError, KeyError, IndexError, TypeError))
                               else "error")
                attempt["reason"] = last_status
                attempt["error_category"] = type(error).__name__
            failed_attempts += 1
            if failed_attempts < max_attempts:
                retry_after_failure = True
                time.sleep(3 if deadline is None else max(0.0, min(3, deadline - time.perf_counter())))

        error = RuntimeError(
            f"LLM request failed after {len(attempts)} attempt(s): "
            f"{last_error or 'unknown error'}"
        )
        error.metadata = accounting()
        error.discarded_attempts = error.metadata["discarded_attempts"]
        error.api_attempts = error.metadata["api_attempts"]
        error.status = last_status
        raise error


# =============================================================================
# 2. LLM Output Normalization & Cluster-Level Structured Road Intent Subsystem
# =============================================================================
def _coerce_llm_content_to_text(content):
    """Normalize various content formats returned by providers into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if "content" in content:
            return _coerce_llm_content_to_text(content["content"])
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = [_coerce_llm_content_to_text(item) for item in content]
        parts = [p for p in parts if p]
        return "\n".join(parts).strip() if parts else json.dumps(content, ensure_ascii=False)
    return str(content).strip()


from utils.cluster_road_intents import (
    _cluster_bounded_shortest_path,
    _cluster_complete_road_context,
    _cluster_hard_time_guard,
    _cluster_road_intent_run,
    _cluster_road_intent_settings,
    _cluster_track_road_graph,
    build_cluster_track_context,
    build_cluster_track_prompt,
    execute_cluster_track_groups,
    parse_cluster_track_groups,
)

