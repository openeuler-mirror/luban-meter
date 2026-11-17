"""Malformed ShareGPT speakers must not discard otherwise usable turns."""

import copy
import json
from types import SimpleNamespace

import pytest

from luban_meter.data import DatasetLoader
from luban_meter.data.projections import extract_conversation, split_turns


@pytest.mark.parametrize("speaker", [[], ["human"], {}, {"role": "human"}])
@pytest.mark.parametrize("strip", [False, True])
def test_non_string_speaker_preserves_valid_turns(tmp_path, speaker, strip):
    record = {"conversations": [
        {"from": "system", "value": " rules "},
        {"from": speaker, "value": "must not become a user message"},
        {"from": "human", "value": " first "},
        {"from": "gpt", "value": "reference answer"},
        {"from": "human", "value": " second "},
    ]}
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    original = copy.deepcopy(record)
    loaded = DatasetLoader(path, format="sharegpt").load()
    expected = [
        [{"role": "system", "content": " rules "},
         {"role": "user", "content": " first "}],
        [{"role": "user", "content": " second "}],
    ]
    if strip:
        for turn in expected:
            for message in turn:
                message["content"] = message["content"].strip()
    projected = extract_conversation(loaded[0], strip=strip)
    assert split_turns(projected) == expected
    assert loaded == [original]


@pytest.mark.parametrize("speaker", [None, True, 1, 1.5, "unknown", ""])
def test_other_unknown_speakers_remain_ignored(speaker):
    record = {"conversations": [
        {"from": speaker, "value": "ignored"},
        {"from": "human", "value": "question"},
    ]}
    assert extract_conversation(record) == [
        {"role": "user", "content": "question"},
    ]


def test_non_object_turn_still_rejected():
    with pytest.raises(ValueError, match="turn must be an object"):
        extract_conversation({"conversations": ["invalid"]})


def test_multi_turn_prepare_keeps_usable_conversation(tmp_path):
    from luban_meter.benchmarking.generation_performance import (
        multi_turn_serving,
    )
    from luban_meter.core.planning import build_plan

    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([{"conversations": [
        {"from": {"role": "human"}, "value": "ignored"},
        {"from": "human", "value": "first"},
        {"from": "gpt", "value": "reference"},
        {"from": "human", "value": "second"},
    ]}]), encoding="utf-8")
    plan = build_plan({
        "scenario": "multi_turn_serving", "model": "test-model",
        "data": "sharegpt", "dataset_path": str(path),
        "multi_num_sessions": 1, "multi_max_turns": 2,
        "profile": "concurrent", "max_concurrency": 1,
    })
    context = SimpleNamespace(parameters=plan["tasks"][0]["cases"][0][
        "parameters"
    ], total_samples=0)
    benchmark = multi_turn_serving.MultiTurnServingBenchmark()
    try:
        benchmark.prepare(context)
        assert context.total_samples == 1
        assert benchmark.sessions[0]["turns"] == [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
        assert benchmark.filter_facts["planned_requests_total"] == 2
    finally:
        benchmark.close()
