"""Fixed-length token ID inputs for controlled prefix-cache experiments."""

import random
import secrets
from fractions import Fraction

from luban_meter.api import GenerationPerformanceBenchmark, Parameter
from luban_meter.benchmarking.benchmark_parameters import HTTP
from luban_meter.core.benchmark import benchmarks
from luban_meter.log import (
    add_log_secrets, get_logger, log_context, redacted_exception_info,
)
from luban_meter.runtime.http_client import (
    http_settings, iter_response_lines,
)
from luban_meter.runtime.case_recovery import save_setup_diagnostic
from .common.http_client import request_headers
from .common.cache_usage import parse_cache_usage
from .common.streaming import collect_completion_stream

from .common.tokenization import check_context, derived_seed, load_tokenizer
from .generation_benchmarks import OnlineServingBenchmark, thresholds


logger = get_logger(__name__)


def prefix_layout(parameters):
    """Quantize an input-token target using the full-block logits limit."""
    total = parameters["input_tokens"]
    block = parameters["kv_cache_block_size"]
    target = Fraction(str(parameters["prefix_cache_target"]))
    desired = total * target
    shared = (min(desired, total - 1) // block) * block
    if target and not shared:
        raise ValueError("positive prefix_cache_target rounds to zero tokens")
    if parameters["prefix_cache_alignment"] == "strict" and shared != desired:
        raise ValueError("prefix_cache_target cannot be represented exactly")
    ratio = Fraction(shared, total)
    return {
        "target_rate": float(target),
        "theoretical_shared_ratio": float(ratio),
        "construction_deviation": float(ratio - target),
        "shared_prefix_tokens": shared,
        "suffix_tokens": total - shared,
    }


def unique_segment(vocabulary, size, block, rng, used):
    """Reserve a distinct leading block, or all tokens of a short segment."""
    width = min(size, block)
    for _ in range(64):
        head = tuple(rng.choices(vocabulary, k=width))
        if head not in used:
            used.add(head)
            return list(head) + rng.choices(vocabulary, k=size - width)
    raise ValueError(
        f"cannot generate distinct {width}-token head after 64 attempts "
        f"(vocabulary={len(vocabulary)}, reserved={len(used)})"
    )


@benchmarks.register("prefix_cache")
class PrefixCacheBenchmark(OnlineServingBenchmark):
    """Prepare token IDs directly and reuse the Completions transport."""

    capabilities = OnlineServingBenchmark.capabilities | {"prefix_cache_usage"}
    default_metrics = ("streaming_performance", "prefix_cache_hit_rate")
    profile_modes = frozenset({"requests"})
    parameters = {
        **HTTP,
        **{
            key: OnlineServingBenchmark.parameters[key]
            for key in (
                "profile", "num_requests", "tokenizer", "input_tokens",
                "output_tokens", "ignore_eos", "max_model_len", "stop",
                "thinking_metrics",
                "goodput", "circuit_breaker", "max_error_rate",
            )
        },
        "endpoint": Parameter(str, "/v1/completions",
                              choices=("/v1/completions",)),
        "length_policy": Parameter(str, "strict", choices=("strict",)),
        "prefix_cache_target": Parameter(
            float, required=True, minimum=0, maximum=1, matrix=True,
        ),
        "prefix_count": Parameter(int, 1, minimum=1, matrix=True),
        "kv_cache_block_size": Parameter(int, required=True, minimum=1),
        "prefix_cache_runtime_tolerance": Parameter(
            float, 0.0, minimum=0, maximum=1,
        ),
        "prefix_cache_environment": Parameter(
            str, "unknown", choices=("unknown", "controlled", "interfered"),
            help="Legacy annotation; ignored for scoring and ranking",
        ),
        "prefix_cache_warmup_policy": Parameter(
            str, "observe", choices=("strict", "observe"),
        ),
        "prefix_cache_alignment": Parameter(
            str, "floor", choices=("floor", "strict"),
        ),
    }

    @classmethod
    def parameter_schema(cls):
        # Inherit runtime controls, but opt in to fixed-length inputs only.
        return GenerationPerformanceBenchmark.parameter_schema() | (
            cls.parameters
        )

    @classmethod
    def validate(cls, parameters):
        http_settings(parameters)
        thresholds(parameters["goodput"], ("ttft", "tpot", "e2el"))
        if parameters["request_params"]:
            raise ValueError("prefix_cache requires empty request_params")
        if parameters["prefix_count"] > parameters["num_requests"]:
            raise ValueError("prefix_count must not exceed num_requests")
        prefix_layout(parameters)
        check_context(parameters["input_tokens"], parameters["output_tokens"],
                      parameters["max_model_len"])

    def _prepare_inputs(self, context):
        p = self.p
        self.records = None
        self.tokenizer = load_tokenizer(p)
        special = set(self.tokenizer.all_special_ids)
        vocabulary = sorted({
            token for token in self.tokenizer.get_vocab().values()
            if type(token) is int and 0 <= token < self.tokenizer.vocab_size
            and token not in special
        })
        if not vocabulary:
            raise ValueError("no usable non-special token IDs")
        layout = prefix_layout(p)
        self.input_metadata = {"prefix_cache": layout}
        block = p["kv_cache_block_size"]
        shared = layout["shared_prefix_tokens"]
        roots = set()
        suffix_heads = set() if shared else roots

        def segment(phase, index, purpose, length, used):
            rng = random.Random(derived_seed(p["seed"], phase, index, purpose))
            return unique_segment(vocabulary, length, block, rng, used)

        self.prefixes = [
            segment("formal", i, "prefix", shared, roots) if shared else []
            for i in range(p["prefix_count"])
        ]
        self.generated = {"formal": [], "warmup": []}
        for phase, count in (
            ("formal", p["num_requests"]), ("warmup", p["num_warmups"]),
        ):
            for index in range(count):
                if phase == "formal":
                    group = index % len(self.prefixes)
                    prefix = self.prefixes[group]
                    suffix = segment(phase, index, "suffix",
                                     layout["suffix_tokens"], suffix_heads)
                    prompt = prefix + suffix
                else:
                    # Ordinary warmups use separate roots and RNG streams.
                    group = None
                    prompt = segment(phase, index, "input",
                                     p["input_tokens"], roots)
                self.generated[phase].append({
                    "index": index, "prefix_id": group, "prompt": prompt,
                    "shared_prefix_tokens": shared if group is not None else 0,
                    "target_input_tokens": p["input_tokens"],
                    "local_input_tokens": len(prompt),
                    "target_output_tokens": p["output_tokens"],
                })
        self._salts = [secrets.token_urlsafe(32) for _ in range(3)]
        add_log_secrets(self._salts)
        self.cache_inputs = {"seed": [], "probe": [], "verify": []}
        if shared:
            for stage in self.cache_inputs:
                for i, prefix in enumerate(self.prefixes):
                    suffix = [] if stage == "seed" else segment(
                        stage, i, "suffix", layout["suffix_tokens"],
                        suffix_heads,
                    )
                    self.cache_inputs[stage].append(prefix + suffix)
        else:
            self.cache_inputs = {"capability": [
                segment("capability", 0, "input", p["input_tokens"], set())
            ]}
        context.total_samples = p["num_requests"]


    def _decorate_record(self, record, sample, context):
        if context.phase == "formal":
            usage = parse_cache_usage(record.get("usage"))
            rate = None
            if (record.get("status") == "success"
                    and usage.state == "valid" and usage.prompt_tokens):
                rate = usage.cached_tokens / usage.prompt_tokens
            record["prefix_cache"] = {
                "prefix_id": sample["prefix_id"],
                "shared_prefix_tokens": sample["shared_prefix_tokens"],
                "usage_state": usage.state, "reason": usage.reason,
                "prompt_tokens": usage.prompt_tokens,
                "cached_tokens": usage.cached_tokens,
                "hit_rate": rate,
            }
        return record

    def _request_payload(self, payload, context):
        salt = self._salts[1 if context.phase == "warmup" else 0]
        return {**payload, "cache_salt": salt}

    def _execute_request(self, **kwargs):
        try:
            record = super()._execute_request(
                **kwargs, exception_log_info=redacted_exception_info,
            )
        except Exception as exc:
            logger.error("Prefix cache request failed: index=%s",
                         kwargs.get("request_index"),
                         exc_info=redacted_exception_info(exc))
            raise RuntimeError("prefix cache transport failed") from None

        def scrub(value):
            if isinstance(value, str):
                for salt in self._salts:
                    value = value.replace(salt, "[redacted]")
                return value
            if isinstance(value, list):
                return [scrub(item) for item in value]
            if isinstance(value, dict):
                return {scrub(k): scrub(v) for k, v in value.items()}
            return value

        for field in ("error", "connection_cleanup_error"):
            if field in record:
                record[field] = {"type": "transport_error"}
        return scrub(record)

    def after_warmup(self, context):
        context.phase = "cache_warmup"
        diagnostic = {
            "status": "running", "seed_completed": 0,
            "probe_completed": 0, "verify_completed": 0,
            "warmup_policy": self.p["prefix_cache_warmup_policy"],
            "warmup_completed": False, "warmup_fully_matched": False,
            "warmup_under_hit_prefixes": 0, "warmup_summary": {},
        }
        fully_matched = True
        under_hit_prefixes = set()
        self.input_metadata["cache_warmup"] = diagnostic
        shared = self.input_metadata["prefix_cache"]["shared_prefix_tokens"]
        logger.info("Cache warmup started: policy=%s prefixes=%s",
                    diagnostic["warmup_policy"], self.p["prefix_count"])
        for stage, prompts in self.cache_inputs.items():
            logger.info("Cache warmup stage started: stage=%s requests=%s",
                        stage, len(prompts))
            for index, prompt in enumerate(prompts):
                diagnostic.update(phase=stage, prefix_id=index)
                for key in ("cached_tokens", "prompt_tokens",
                            "expected_cached_tokens", "reason"):
                    diagnostic.pop(key, None)
                expected = None if stage == "seed" else shared
                if expected is not None:
                    diagnostic["expected_cached_tokens"] = expected
                save_setup_diagnostic(
                    context.directory, context.case_id, diagnostic,
                )
                reason = None
                try:
                    if context.stopped():
                        raise InterruptedError()
                    result = context.request(
                        lambda ticket: self._cache_request(
                            prompt, stage, context, index, ticket.index,
                        )
                    )
                    diagnostic.update(result)
                    reason = result.get("reason")
                    if context.stopped():
                        reason = "interrupted"
                    if expected is not None:
                        actual = result.get("cached_tokens")
                        state = (
                            reason if reason in ("missing", "invalid")
                            else "unavailable" if reason
                            else "matched" if actual == expected
                            else "under_hit" if actual < expected
                            else "over_hit"
                        )
                        diagnostic["warmup_summary"].setdefault(
                            stage, [],
                        ).append({
                            "prefix_id": index,
                            "expected_cached_tokens": expected,
                            "actual_cached_tokens": actual,
                            "deviation_tokens": (
                                actual - expected
                                if type(actual) is int else None
                            ),
                            "validation_state": state,
                        })
                        if state == "under_hit":
                            under_hit_prefixes.add(index)
                        diagnostic["warmup_under_hit_prefixes"] = len(
                            under_hit_prefixes
                        )
                        if state != "matched":
                            fully_matched = False
                        if not reason and state != "matched":
                            if diagnostic["warmup_policy"] == "strict":
                                reason = "cache_mismatch"
                except Exception as exc:
                    logger.error(
                        "Cache warmup failed: stage=%s prefix_id=%s",
                        stage, index, exc_info=redacted_exception_info(exc),
                    )
                    reason = (
                        "interrupted" if context.stopped()
                        else "transport_error"
                    )
                if reason:
                    logger.error(
                        "Cache warmup rejected: stage=%s prefix_id=%s "
                        "reason=%s expected=%s actual=%s",
                        stage, index, reason, expected,
                        diagnostic.get("cached_tokens"),
                    )
                    diagnostic.update(status="failed", reason=reason)
                    save_setup_diagnostic(
                        context.directory, context.case_id, diagnostic,
                    )
                    raise RuntimeError("cache warmup failed") from None
                if stage != "capability":
                    diagnostic[stage + "_completed"] += 1
        diagnostic.update(
            warmup_completed=True, warmup_fully_matched=fully_matched,
            status=("observed_mismatch" if not fully_matched else
                    "success" if shared else "not_applicable"),
        )
        logger.info(
            "Cache warmup completed: policy=%s fully_matched=%s under_hit=%s",
            diagnostic["warmup_policy"], fully_matched,
            len(under_hit_prefixes),
        )
        save_setup_diagnostic(context.directory, context.case_id, diagnostic)

    def _cache_request(self, prompt, stage, context, prefix_id, request_id):
        with log_context(phase="cache_warmup", request=request_id):
            return self._send_cache_request(prompt, stage, context, prefix_id)

    def _send_cache_request(self, prompt, stage, context, prefix_id):
        payload = {
            "model": self.p["model"], "prompt": prompt,
            "add_special_tokens": False, "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 1, "ignore_eos": True, "temperature": 0,
            "cache_salt": self._salts[2 if stage == "capability" else 0],
        }
        try:
            with self.client.stream(
                "POST", self.p["base_url"].rstrip("/") + "/v1/completions",
                json=payload, headers=request_headers(self.p["api_key"]),
                timeout=self.p["request_timeout"],
            ) as response:
                response.raise_for_status()

                def lines():
                    for line in iter_response_lines(response):
                        if context.stopped():
                            raise InterruptedError()
                        yield line

                observation = collect_completion_stream(
                    lines(), require_output=False,
                )
            parsed = parse_cache_usage(observation.usage)
            result = {}
            for key in ("prompt_tokens", "cached_tokens"):
                value = getattr(parsed, key)
                if value is not None:
                    result[key] = value
            if parsed.state != "valid":
                result["reason"] = parsed.state
            elif parsed.prompt_tokens != len(prompt):
                result["reason"] = "input_length"
            elif (observation.output_tokens != 1
                  or observation.finish_reason not in ("stop", "length")):
                result["reason"] = "completion"
            return result
        except Exception as exc:
            logger.error(
                "Cache warmup request failed: stage=%s prefix_id=%s",
                stage, prefix_id, exc_info=redacted_exception_info(exc),
            )
            kind = type(exc).__name__
            allowed = {
                "ReadTimeout", "HTTPStatusError", "RuntimeError",
                "ValueError", "JSONDecodeError",
            }
            return {"reason": kind if kind in allowed else "transport_error"}
