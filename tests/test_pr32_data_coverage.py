"""PR !32 loading and projection capabilities through the new interface."""

import json
from copy import deepcopy

import pytest

from luban_meter.data import (
    DatasetLoader,
    extract_prompts,
    extract_traces,
    select_columns,
)
from luban_meter.data.engine import select_few_shot


@pytest.mark.parametrize(
    "key", [None, "data", "rows", "examples", "questions"]
)
def test_json_containers_and_complete_records(tmp_path, key):
    records = [{"prompt": "hello", "answer": "world", "subject": "math"}]
    path = tmp_path / "input.json"
    path.write_text(json.dumps({key: records} if key else records))
    loaded = DatasetLoader(path).load()
    assert loaded == records
    assert select_columns(loaded, ["prompt"]) == [{"prompt": "hello"}]
    assert loaded == records
    assert select_few_shot(loaded, "subject", "math", 1) == records


def test_burstgpt_loading_and_trace_projection(tmp_path):
    path = tmp_path / "trace.csv"
    path.write_text("timestamp,input_token,output_token,tag\n1.5,16,8,kept\n")
    records = DatasetLoader(path, format="burstgpt").load()
    original = deepcopy(records)
    assert records[0]["tag"] == "kept"
    assert records[0]["timestamp"] == "1.5"
    assert extract_traces(records) == [
        {
            "offset_s": 1.5,
            "input_tokens": 16,
            "output_tokens": 8,
        }
    ]
    assert records == original


def test_sharegpt_and_named_prompt_projection(tmp_path):
    rows = [
        {"conversations": [{"from": "gpt", "value": "answer"}]},
        {
            "conversations": [
                {"from": "human", "value": " first "},
                {"from": "human", "value": "second"},
            ]
        },
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(rows))
    records = DatasetLoader(path, format="sharegpt").load()
    assert len(records) == 2
    assert extract_prompts(records) == ["first"]
    assert records == rows
    assert extract_prompts([{"question": " custom "}], "question") == [
        "custom"
    ]


def test_prompt_sampling_uses_full_population():
    records = [{"prompt": str(index)} for index in range(100)]
    prompts = extract_prompts(records, max_samples=5, shuffle=True, seed=7)
    assert prompts == extract_prompts(
        records, max_samples=5, shuffle=True, seed=7
    )
    assert len(prompts) == 5
    assert any(int(prompt) >= 5 for prompt in prompts)


def test_projection_errors():
    with pytest.raises(ValueError):
        extract_prompts([{}])
    with pytest.raises(ValueError):
        extract_traces([{"offset_s": True}])
    with pytest.raises(ValueError):
        extract_prompts([{"prompt": "valid"}], max_samples=0)
