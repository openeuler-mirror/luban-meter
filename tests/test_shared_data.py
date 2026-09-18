"""The same local record engine feeds both benchmark categories."""

import json

import pytest

from luban_meter.data import DatasetLoader
from luban_meter.data.sources import DataSource, sources


@pytest.mark.parametrize("kind", ["json", "jsonl"])
def test_records_preserve_all_fields_and_seed(kind, tmp_path):
    rows = [
        {"prompt": str(i), "answer": i, "tests": ["assert True"]}
        for i in range(20)
    ]
    path = tmp_path / ("data." + kind)
    path.write_text(
        json.dumps(rows)
        if kind == "json"
        else "\n".join(json.dumps(r) for r in rows)
    )
    loader = DatasetLoader(
        path, format=kind, max_samples=5, shuffle=True, seed=10
    )
    first = loader.load()
    assert first == loader.load()
    assert len(first) == 5
    assert first != rows[:5]
    assert all(set(r) == {"prompt", "answer", "tests"} for r in first)


def test_sharegpt_preserves_complete_conversations(tmp_path):
    path = tmp_path / "data.json"
    rows = [
        {"id": "a", "conversations": [{"from": "gpt", "value": "x"}]},
        {
            "id": "b",
            "conversations": [
                {"from": "human", "value": " hello "},
                {"from": "human", "value": "second"},
            ],
        },
    ]
    path.write_text(json.dumps(rows))
    result = DatasetLoader(path, format="sharegpt").load()
    assert result == rows


@pytest.mark.parametrize("kind", ["json", "jsonl"])
def test_invalid_record_is_not_silently_discarded(tmp_path, kind):
    path = tmp_path / ("data." + kind)
    path.write_text("[{}, 7]" if kind == "json" else "{}\n7\n")
    with pytest.raises(TypeError):
        DatasetLoader(path, format=kind).load()


def test_unicode_line_separator_in_record(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "first\u2028second"}\n')
    assert DatasetLoader(path).load()[0]["text"] == ("first\u2028second")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DatasetLoader(tmp_path / "absent.jsonl").load()


@pytest.mark.parametrize("suffix", ["csv", "CSV"])
def test_csv_inference_preserves_columns_and_quoted_text(tmp_path, suffix):
    path = tmp_path / f"records.{suffix}"
    path.write_text('prompt,answer,metadata\n"hello, world",42,extra\n')
    assert DatasetLoader(path).load() == [
        {"prompt": "hello, world", "answer": "42", "metadata": "extra"}
    ]


@pytest.mark.parametrize("contents", ["a,a\n1,2", "a,b\n1", "a\n1,2"])
def test_csv_rejects_ambiguous_or_incomplete_records(tmp_path, contents):
    path = tmp_path / "data.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        DatasetLoader(path).load()


def test_explicit_format_overrides_suffix(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('{"id": 1}\n{"id": 2}\n')
    assert DatasetLoader(path, format="jsonl").load() == [{"id": 1}, {"id": 2}]


def test_explicit_base_dir_does_not_use_working_directory(
    tmp_path, monkeypatch
):
    directory = tmp_path / "config"
    directory.mkdir()
    (tmp_path / "records.json").write_text('[{"id": "wrong"}]')
    (directory / "records.json").write_text('[{"id": "right"}]')
    monkeypatch.chdir(tmp_path)
    loader = DatasetLoader("records.json", base_dir=directory)
    assert loader.load() == [{"id": "right"}]
    (directory / "records.json").unlink()
    with pytest.raises(FileNotFoundError):
        loader.load()


@pytest.mark.parametrize(
    "options",
    [
        {"max_samples": 0},
        {"max_samples": True},
        {"max_samples": 1.5},
        {"shuffle": "yes"},
        {"seed": True},
        {"seed": -1},
        {"format": ""},
        {"format": "unknown"},
    ],
)
def test_invalid_loader_options(options):
    with pytest.raises(ValueError):
        DatasetLoader("data.json", **options)


def test_custom_source_keeps_loader_sampling(tmp_path):
    @sources.register("fixture_records")
    class FixtureSource(DataSource):
        def read(self, path):
            return [{"answer": value} for value in range(10)]

    path = tmp_path / "custom.input"
    path.touch()
    loader = DatasetLoader(
        path, format="fixture_records", max_samples=3, shuffle=True, seed=7
    )
    assert loader.load() == loader.load()
    assert len(loader.load()) == 3
    assert loader.load() != [{"answer": value} for value in range(3)]


def test_quality_main_and_few_shot_share_loader(tmp_path):
    from types import SimpleNamespace

    from luban_meter.benchmarking.model_service_quality import (
        quality_benchmarks,
    )
    from luban_meter.core.planning import build_plan

    main = tmp_path / "questions.jsonl"
    main.write_text(
        json.dumps({"question": "1+1?", "answer": "#### 2", "source": "main"})
        + "\n"
    )
    examples = tmp_path / "examples.csv"
    examples.write_text("question,answer,source\n2+2?,#### 4,few-shot\n")
    plan = build_plan(
        {
            "scenario": "gsm8k",
            "model": "test",
            "dataset_path": str(main),
            "few_shot_path": str(examples),
            "few_shot": 1,
        }
    )
    context = SimpleNamespace(
        parameters=plan["tasks"][0]["cases"][0]["parameters"],
        request=lambda *args, **kwargs: None,
    )
    scenario = quality_benchmarks.GSM8KBenchmark()
    scenario.prepare(context)
    assert scenario.records[0]["source"] == "main"
    assert scenario.few_shot[0]["source"] == "few-shot"
    assert scenario.few_shot[0]["reference_number"] == 4
    assert context.total_samples == 1
