"""Tests for prepare_lcsts.py data processing script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "src"
    / "luban_meter"
    / "benchmark"
    / "inference"
    / "scripts"
    / "prepare_lcsts.py"
)

MODULE_COUNT = 0


def load_script_module():
    """Load prepare_lcsts.py as a module."""
    global MODULE_COUNT
    MODULE_COUNT += 1
    spec = importlib.util.spec_from_file_location(
        f"prepare_lcsts_test_{MODULE_COUNT}",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return load_script_module()


# ---- strip_instruction_prefix ----


def test_strip_prefix_present(mod) -> None:
    """Strip prefix when present."""
    prefix = mod.LCSTS_INSTRUCTION_PREFIX
    text = prefix + "\u4eca\u5929\u5929\u6c14\u5f88\u597d"
    result = mod.strip_instruction_prefix(text)
    assert result == "\u4eca\u5929\u5929\u6c14\u5f88\u597d"


def test_strip_prefix_absent(mod) -> None:
    """Return original text when prefix not present."""
    text = "\u4eca\u5929\u5929\u6c14\u5f88\u597d"
    result = mod.strip_instruction_prefix(text)
    assert result == text


def test_strip_prefix_empty_after(mod) -> None:
    """Return empty string if only prefix present."""
    result = mod.strip_instruction_prefix(
        mod.LCSTS_INSTRUCTION_PREFIX
    )
    assert result == ""


# ---- load_jsonl ----


def test_load_jsonl_valid(tmp_path: Path, mod) -> None:
    """Load valid JSONL file."""
    data_file = tmp_path / "data.json"
    records = [
        {
            "input": "prefix text",
            "output": "summary",
            "id": 0,
        },
        {
            "input": "prefix text2",
            "output": "summary2",
            "id": 1,
        },
    ]
    data_file.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in records
        ),
        encoding="utf-8",
    )
    loaded = mod.load_jsonl(data_file)
    assert len(loaded) == 2
    assert loaded[0]["input"] == "prefix text"
    assert loaded[1]["output"] == "summary2"


def test_load_jsonl_skips_empty_lines(
    tmp_path: Path, mod
) -> None:
    """Skip empty lines in JSONL."""
    data_file = tmp_path / "data.json"
    data_file.write_text(
        '{"input": "a", "output": "b", "id": 0}\n'
        "\n"
        '{"input": "c", "output": "d", "id": 1}\n',
        encoding="utf-8",
    )
    loaded = mod.load_jsonl(data_file)
    assert len(loaded) == 2


def test_load_jsonl_file_not_found(mod) -> None:
    """Raise FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        mod.load_jsonl(Path("/nonexistent/file.json"))


def test_load_jsonl_empty_file(
    tmp_path: Path, mod
) -> None:
    """Raise ValueError for empty file."""
    data_file = tmp_path / "empty.json"
    data_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        mod.load_jsonl(data_file)


def test_load_jsonl_invalid_json(
    tmp_path: Path, mod
) -> None:
    """Raise ValueError for invalid JSON."""
    data_file = tmp_path / "bad.json"
    data_file.write_text(
        "not valid json", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        mod.load_jsonl(data_file)


def test_load_jsonl_non_object(
    tmp_path: Path, mod
) -> None:
    """Raise ValueError for non-object records."""
    data_file = tmp_path / "arr.json"
    data_file.write_text(
        '[1, 2, 3]', encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="must be an object"
    ):
        mod.load_jsonl(data_file)


# ---- validate_record ----


def test_validate_record_valid(mod) -> None:
    """Valid record passes validation."""
    record = {
        "input": mod.LCSTS_INSTRUCTION_PREFIX
        + "\u6587\u672c",
        "output": "\u6458\u8981",
        "id": 0,
    }
    mod.validate_record(record, 1, "test.json")


def test_validate_record_missing_input(mod) -> None:
    """Raise ValueError for missing input."""
    record = {"output": "summary", "id": 0}
    with pytest.raises(
        ValueError, match="missing 'input'"
    ):
        mod.validate_record(record, 1, "test.json")


def test_validate_record_missing_output(mod) -> None:
    """Raise ValueError for missing output."""
    record = {"input": "text", "id": 0}
    with pytest.raises(
        ValueError, match="missing 'output'"
    ):
        mod.validate_record(record, 1, "test.json")


def test_validate_record_non_string_input(mod) -> None:
    """Raise ValueError for non-string input."""
    record = {"input": 123, "output": "summary"}
    with pytest.raises(
        ValueError, match="'input' must be string"
    ):
        mod.validate_record(record, 1, "test.json")


def test_validate_record_empty_content(mod) -> None:
    """Raise ValueError for empty content after strip."""
    record = {
        "input": mod.LCSTS_INSTRUCTION_PREFIX,
        "output": "\u6458\u8981",
    }
    with pytest.raises(
        ValueError, match="'input' is empty"
    ):
        mod.validate_record(record, 1, "test.json")


def test_validate_record_empty_output(mod) -> None:
    """Raise ValueError for empty output."""
    record = {
        "input": mod.LCSTS_INSTRUCTION_PREFIX
        + "\u6587\u672c",
        "output": "",
    }
    with pytest.raises(
        ValueError, match="'output' is empty"
    ):
        mod.validate_record(record, 1, "test.json")


# ---- convert_record ----


def test_convert_record(mod) -> None:
    """Convert record with prefix stripping."""
    record = {
        "input": mod.LCSTS_INSTRUCTION_PREFIX
        + "\u4eca\u5929\u5929\u6c14\u5f88\u597d",
        "output": "\u5929\u6c14\u597d",
        "id": 0,
    }
    result = mod.convert_record(record, 5, "lcsts-test")
    assert result["id"] == "lcsts-test-5"
    assert result["content"] == "\u4eca\u5929\u5929\u6c14\u5f88\u597d"
    assert result["abst"] == "\u5929\u6c14\u597d"


def test_convert_record_no_prefix(mod) -> None:
    """Convert record without prefix."""
    record = {
        "input": "\u4eca\u5929\u5929\u6c14\u5f88\u597d",
        "output": "\u5929\u6c14\u597d",
        "id": 0,
    }
    result = mod.convert_record(record, 0, "lcsts-fs")
    assert result["id"] == "lcsts-fs-0"
    assert result["content"] == "\u4eca\u5929\u5929\u6c14\u5f88\u597d"


# ---- validate_dataset_integrity ----


def test_validate_integrity_correct(mod) -> None:
    """Pass with correct count."""
    records = [{"a": 1}] * 725
    mod.validate_dataset_integrity(
        records, 725, "test.json"
    )


def test_validate_integrity_wrong(mod) -> None:
    """Raise ValueError for wrong count."""
    records = [{"a": 1}] * 100
    with pytest.raises(
        ValueError, match="expected 725"
    ):
        mod.validate_dataset_integrity(
            records, 725, "test.json"
        )


# ---- validate_all_records ----


def test_validate_all_valid(mod) -> None:
    """Pass with all valid records."""
    records = [
        {
            "input": mod.LCSTS_INSTRUCTION_PREFIX
            + f"\u6587\u672c{i}",
            "output": f"\u6458\u8981{i}",
        }
        for i in range(3)
    ]
    mod.validate_all_records(records, "test.json")


def test_validate_all_one_invalid(mod) -> None:
    """Raise ValueError if any record is invalid."""
    records = [
        {
            "input": mod.LCSTS_INSTRUCTION_PREFIX
            + "\u6587\u672c",
            "output": "\u6458\u8981",
        },
        {"output": "\u6458\u8981"},  # missing input
    ]
    with pytest.raises(
        ValueError, match="missing 'input'"
    ):
        mod.validate_all_records(records, "test.json")


# ---- write_jsonl ----


def test_write_jsonl(tmp_path: Path, mod) -> None:
    """Write records to JSONL file."""
    records = [
        {"id": "a", "content": "x", "abst": "y"},
        {"id": "b", "content": "p", "abst": "q"},
    ]
    out_file = tmp_path / "out.jsonl"
    mod.write_jsonl(records, out_file)
    lines = out_file.read_text(
        encoding="utf-8"
    ).strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "a"
    assert json.loads(lines[1])["content"] == "p"


# ---- End-to-end with real data ----


def test_real_data_integrity(mod) -> None:
    """Verify the bundled test.jsonl has correct count."""
    data_path = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "data"
        / "lcsts"
        / "test.jsonl"
    )
    if not data_path.is_file():
        pytest.skip("test.jsonl not generated yet")
    records = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert len(records) == mod.EXPECTED_TEST_COUNT


def test_real_data_fields(mod) -> None:
    """Verify all records have required fields."""
    data_path = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "data"
        / "lcsts"
        / "test.jsonl"
    )
    if not data_path.is_file():
        pytest.skip("test.jsonl not generated yet")
    with data_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert "id" in record, f"line {idx}"
            assert "content" in record, f"line {idx}"
            assert "abst" in record, f"line {idx}"
            assert record["id"].startswith("lcsts-test-")
            assert len(record["content"]) > 0
            assert len(record["abst"]) > 0


def test_real_data_no_prefix(mod) -> None:
    """Verify prefix was stripped from content."""
    data_path = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "data"
        / "lcsts"
        / "test.jsonl"
    )
    if not data_path.is_file():
        pytest.skip("test.jsonl not generated yet")
    prefix = mod.LCSTS_INSTRUCTION_PREFIX
    with data_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            assert not record["content"].startswith(
                prefix
            ), f"line {idx} still has prefix"


def test_real_data_unique_ids(mod) -> None:
    """Verify all IDs are unique."""
    data_path = (
        ROOT
        / "src"
        / "luban_meter"
        / "benchmark"
        / "inference"
        / "data"
        / "lcsts"
        / "test.jsonl"
    )
    if not data_path.is_file():
        pytest.skip("test.jsonl not generated yet")
    ids = []
    with data_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                record = json.loads(line)
                ids.append(record["id"])
    assert len(ids) == len(set(ids))


def test_end_to_end_local_source(
    tmp_path: Path, mod
) -> None:
    """End-to-end test with local source files."""
    # Create fake source data
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    prefix = mod.LCSTS_INSTRUCTION_PREFIX
    test_records = [
        {
            "input": prefix + f"\u6587\u7ae0{i}",
            "output": f"\u6458\u8981{i}",
            "id": 0,
        }
        for i in range(mod.EXPECTED_TEST_COUNT)
    ]
    train_records = [
        {
            "input": prefix + f"\u8bad\u7ec3{i}",
            "output": f"\u6458\u8981{i}",
            "id": 0,
        }
        for i in range(mod.EXPECTED_TRAIN_COUNT)
    ]
    (source_dir / "test.json").write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in test_records
        ),
        encoding="utf-8",
    )
    (source_dir / "train.json").write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in train_records
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "output"
    # Process test split
    count = mod.process_split(
        source_dir / "test.json",
        out_dir / "test.jsonl",
        mod.EXPECTED_TEST_COUNT,
        "lcsts-test",
    )
    assert count == mod.EXPECTED_TEST_COUNT
    # Verify output
    lines = (out_dir / "test.jsonl").read_text(
        encoding="utf-8"
    ).strip().split("\n")
    assert len(lines) == mod.EXPECTED_TEST_COUNT
    first = json.loads(lines[0])
    assert first["id"] == "lcsts-test-0"
    assert first["content"] == "\u6587\u7ae00"
    assert first["abst"] == "\u6458\u89810"
    # Process few-shot
    train_recs = mod.load_jsonl(
        source_dir / "train.json"
    )
    mod.validate_all_records(
        train_recs, "train.json"
    )
    few_shot = mod.convert_records(
        train_recs[:8], "lcsts-fs"
    )
    mod.write_jsonl(
        few_shot, out_dir / "few_shot.jsonl"
    )
    fs_lines = (out_dir / "few_shot.jsonl").read_text(
        encoding="utf-8"
    ).strip().split("\n")
    assert len(fs_lines) == 8
    fs_first = json.loads(fs_lines[0])
    assert fs_first["id"] == "lcsts-fs-0"
    assert fs_first["content"] == "\u8bad\u7ec30"
