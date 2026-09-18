"""Validated trace rows and record-preserving selection."""

import csv
import math
import random
import re


def read_trace(path, *, skip_invalid=False):
    records, rejected = [], []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{path}: trace requires unique named columns")
        for row in reader:
            converted = {}
            error = None
            for field, target in (
                ("timestamp", "offset_s"),
                ("input_token", "input_tokens"),
                ("output_token", "output_tokens"),
            ):
                value = row.get(field)
                try:
                    if field == "timestamp":
                        parsed = float(value)
                        if not math.isfinite(parsed):
                            raise ValueError("must be finite")
                    else:
                        if not isinstance(value, str) or not re.fullmatch(
                            r"[+]?[0-9]+", value.strip()
                        ):
                            raise ValueError("must be an integer")
                        parsed = int(value)
                        if parsed < 1:
                            raise ValueError("must be positive")
                    converted[target] = parsed
                except (ValueError, TypeError) as exc:
                    error = {
                        "file": str(path),
                        "line": reader.line_num,
                        "field": field,
                        "value": value,
                        "reason": str(exc),
                    }
                    break
            if None in row or any(v is None for v in row.values()):
                error = error or {
                    "file": str(path),
                    "line": reader.line_num,
                    "field": "csv_row",
                    "value": row,
                    "reason": "column count does not match header",
                }
            if error:
                if not skip_invalid:
                    raise ValueError(str(error))
                rejected.append(error)
                continue
            records.append(
                {
                    **row,
                    **converted,
                    "source_file": str(path),
                    "source_line": reader.line_num,
                }
            )
    return records, rejected


def select_trace(records, parameters):
    replay = parameters["profile"] == "replay"
    rows = list(records)
    if replay:
        if parameters.get("shuffle"):
            raise ValueError("replay does not allow shuffle")
        rows.sort(key=lambda row: row["offset_s"])
    elif parameters.get("shuffle"):
        seed = parameters.get("data_seed")
        random.Random(parameters["seed"] if seed is None else seed).shuffle(
            rows
        )
    start = parameters.get("trace_start", 0)
    rows = rows[start:]
    if parameters.get("max_samples"):
        rows = rows[: parameters["max_samples"]]
    count = parameters["num_requests"]
    if count > len(rows):
        raise ValueError(
            f"trace contains {len(rows)} usable records, requested {count}"
        )
    rows = rows[:count]
    origin = rows[0]["offset_s"]
    return [
        {
            **row,
            "scheduling": {
                "trace_index": index,
                "offset_s": row["offset_s"] - origin,
            },
        }
        if replay
        else dict(row)
        for index, row in enumerate(rows)
    ]
