"""Arrival time scheduling for generate benchmarks."""

from __future__ import annotations

import random


def schedule_arrival_times(
    *,
    arrival_process: str,
    request_rate: float,
    num_requests: int,
    burstiness: float,
    seed: int,
) -> list[float]:
    """Return cumulative arrival offsets (seconds from t=0).

    - ``constant``: uniform spacing ``1/rate``.
    - ``poisson``: exponential inter-arrival with mean ``1/rate``.
    - ``gamma``: Gamma(shape=burstiness, scale=1/(rate*burstiness))
      intervals.  ``burstiness < 1`` → more bursty; ``> 1`` → more
      uniform.
    """
    rng = random.Random(seed)
    offsets: list[float] = []
    cumulative = 0.0

    if arrival_process == "constant":
        for i in range(num_requests):
            offsets.append(i / request_rate)
    elif arrival_process == "poisson":
        for _ in range(num_requests):
            offsets.append(cumulative)
            cumulative += rng.expovariate(request_rate)
    elif arrival_process == "gamma":
        shape = burstiness
        scale = 1.0 / (request_rate * burstiness)
        for _ in range(num_requests):
            offsets.append(cumulative)
            cumulative += rng.gammavariate(shape, scale)
    else:
        raise ValueError(
            f"arrival_process must be 'constant', 'poisson', or "
            f"'gamma', got {arrival_process!r}"
        )

    return offsets
