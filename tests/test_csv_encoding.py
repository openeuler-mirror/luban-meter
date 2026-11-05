"""UTF-8 signatures must not become dataset column names."""

import pytest

from luban_meter.data import DatasetLoader, extract_prompts


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("header", ["prompt", '"prompt"'])
def test_csv_encoding_preserves_prompt_and_field_contents(
    tmp_path, encoding, header
):
    path = tmp_path / "prompts.csv"
    path.write_text(
        f'{header},answer\r\n"你好,\n世界\ufeff",42\r\n',
        encoding=encoding,
    )
    records = DatasetLoader(path).load()
    assert extract_prompts(records) == ["你好,\n世界\ufeff"]
    assert records == [{"prompt": "你好,\n世界\ufeff", "answer": "42"}]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
@pytest.mark.parametrize("skip_invalid", [False, True])
def test_burstgpt_encoding_preserves_valid_trace(
    tmp_path, encoding, skip_invalid
):
    path = tmp_path / "trace.csv"
    path.write_text(
        '"timestamp",input_token,output_token\r\n1.5,4,2\r\n',
        encoding=encoding,
    )
    diagnostics = []
    records = DatasetLoader(
        path, format="burstgpt", skip_invalid=skip_invalid,
        diagnostics=diagnostics,
    ).load()
    assert records == [{
        "timestamp": "1.5", "input_token": "4", "output_token": "2",
        "offset_s": 1.5, "input_tokens": 4, "output_tokens": 2,
        "source_file": str(path), "source_line": 2,
    }]
    assert diagnostics == []


@pytest.mark.parametrize(
    "contents",
    ["prompt,prompt\nfirst,second\n", ",answer\nx,42\n"],
)
def test_bom_does_not_hide_invalid_csv_headers(tmp_path, contents):
    path = tmp_path / "invalid.csv"
    path.write_text(contents, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        DatasetLoader(path).load()
