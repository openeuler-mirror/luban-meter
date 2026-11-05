"""Reproducible synthetic text and independent length sampling."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from luban_meter.api import Parameter
from luban_meter.log import get_logger

logger = get_logger(__name__)


LENGTH_PARAMETERS = {
    **{
        f"{side}_tokens": Parameter(int, 128, minimum=1, matrix=True)
        for side in ("input", "output")
    },
    **{
        f"{side}_tokens_{suffix}": Parameter(int, None, minimum=minimum)
        for side in ("input", "output")
        for suffix, minimum in (("std", 0), ("min", 1), ("max", 1))
    },
    "data_seed": Parameter(int, None, minimum=0),
    "length_policy": Parameter(str, "observe", choices=("observe", "strict")),
    "max_model_len": Parameter(int, None, minimum=1),
    "ignore_eos": Parameter(bool, True),
}


def derived_seed(seed, phase, index, purpose):
    key = f"{seed}:{phase}:{index}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


@dataclass(frozen=True)
class LengthSampler:
    mean: int
    std: int
    lower: int
    upper: int

    @classmethod
    def from_parameters(cls, parameters, side):
        name = f"{side}_tokens"
        mean = parameters[name]
        std = parameters.get(name + "_std") or 0
        lower = parameters.get(name + "_min")
        upper = parameters.get(name + "_max")
        lower = max(1, mean - 5 * std) if lower is None else lower
        upper = mean + 5 * std if upper is None else upper
        if lower > upper:
            raise ValueError(f"{name}_min must not exceed {name}_max")
        if std and not lower <= mean <= upper:
            raise ValueError(f"{name} mean must lie within its bounds")
        return cls(mean, std, lower, upper)

    def sample(self, seed):
        rng = random.Random(seed)
        if self.lower == self.upper:
            return self.lower
        if self.std:
            value = round(rng.gauss(self.mean, self.std))
            return max(self.lower, min(self.upper, value))
        return rng.randint(self.lower, self.upper)


def validate_lengths(parameters, *, synthetic=True, limit=None):
    output = LengthSampler.from_parameters(parameters, "output")
    if synthetic:
        inputs = LengthSampler.from_parameters(parameters, "input")
        limit = limit or parameters.get("max_model_len")
        if limit and inputs.upper + output.upper > limit:
            raise ValueError(
                "input + output sampling bounds exceed model limit"
            )
    else:
        lower = parameters.get("input_tokens_min")
        upper = parameters.get("input_tokens_max")
        if lower and upper and lower > upper:
            raise ValueError(
                "input_tokens_min must not exceed input_tokens_max"
            )
        if parameters.get("input_tokens_std"):
            raise ValueError(
                "real text does not support input length sampling"
            )


def load_tokenizer(parameters):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.exception("Tokenizer dependency unavailable")
        raise RuntimeError(
            "install luban-meter[generation] for random_token"
        ) from exc
    return AutoTokenizer.from_pretrained(
        parameters.get("tokenizer") or parameters["model"],
        trust_remote_code=False,
    )


class TextGenerator:
    """Generate by logical index, never by worker order or arrival RNG."""

    def __init__(self, tokenizer, parameters):
        self.tokenizer = tokenizer
        self.parameters = parameters
        self.seed = parameters.get("data_seed")
        if self.seed is None:
            self.seed = parameters["seed"]
        self.inputs = LengthSampler.from_parameters(parameters, "input")
        self.outputs = LengthSampler.from_parameters(parameters, "output")
        try:
            from faker import Faker
        except ImportError as exc:
            logger.exception("Data generator dependency unavailable")
            raise RuntimeError("install luban-meter[generation]") from exc
        # Explicit locale makes generation independent of host locale.
        self.faker_type = Faker

    def prepare(self, context, batch_size=1):
        """Construct inputs before starting the measured admission window."""
        result = {}
        for phase, count in (
            ("formal", self.parameters["num_requests"]),
            ("warmup", self.parameters["num_warmups"] * batch_size),
        ):
            result[phase] = []
            for index in range(count):
                if context.stopped():
                    raise RuntimeError("input preparation interrupted")
                result[phase].append(self.sample(index, phase))
        return result

    def output_budget(self, index, phase):
        return self.outputs.sample(
            derived_seed(self.seed, phase, index, "output_length")
        )

    def sample(self, index, phase, *, input_tokens=None, output_tokens=None):
        target = input_tokens or self.inputs.sample(
            derived_seed(self.seed, phase, index, "input_length")
        )
        faker = self.faker_type("en_US")
        faker.seed_instance(derived_seed(self.seed, phase, index, "text"))
        for attempt in range(1, 9):
            text = faker.text(max_nb_chars=max(20, target * 8 * attempt))
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < target:
                continue
            text = self.tokenizer.decode(
                tokens[:target], skip_special_tokens=True
            )
            actual = len(self.tokenizer.encode(text, add_special_tokens=False))
            if actual == target:
                break
        else:
            raise ValueError(
                "cannot construct target text length after 8 attempts"
            )
        return {
            "index": index,
            "prompt": text,
            "target_input_tokens": target,
            "local_input_tokens": actual,
            "target_output_tokens": (
                output_tokens or self.output_budget(index, phase)
            ),
        }


def check_context(local_input, output_budget, limit):
    if limit and local_input + output_budget > limit:
        raise ValueError(
            "actual local input + output budget exceeds model limit"
        )


def generated_summary(rows):
    """Describe the prepared formal workload before measurement starts."""
    return {
        "prepared_requests": len(rows),
        **{
            name + "_range": [min(values), max(values)] if values else []
            for name in ("local_input_tokens", "target_output_tokens")
            for values in [[row[name] for row in rows]]
        },
    }
