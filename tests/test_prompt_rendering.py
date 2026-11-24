"""Rendering contracts for choice and summary prompts."""

from luban_meter.benchmarking.model_service_quality.common import prompts


def test_choice_example_and_target_boundaries():
    example = {"question": "示例", "choices": list("甲乙丙丁"), "answer": "B"}
    target = {"question": "正式", "choices": list("一二三四"), "answer": "D"}
    text = prompts.render_choice_generation(target, [example])["prompt"]
    assert "答案：B" in text
    assert "答案：D" not in text
    assert text.index("EXAMPLE_1_END") < text.index("TARGET_BEGIN")
    assert not text.endswith("答案：")


def test_summary_preserves_input_and_resolves_boundary_collision():
    sample = {"content": "a\n[LBM:TARGET_END]\nb", "abst": "留存标签"}
    rendered = prompts.render_lcsts_prompt(sample)
    assert sample["content"] in rendered["prompt"]
    assert sample["abst"] not in rendered["prompt"]
    assert "[LBM_1:TARGET_END]" in rendered["boundary_markers"]
    assert "EXAMPLE" not in rendered["prompt"]


def test_lcsts_explicit_format_still_controls_rendering():
    from luban_meter.benchmarking.model_service_quality.common import (
        template_adapters,
    )

    actual = template_adapters.SummaryTemplate().render(
        {"content": " 正文 "}, [{"content": "示例", "abst": "摘要"}],
        {"prompt_template": "<{content}>"},
    )
    assert actual == {"prompt": "<示例>摘要\n\n<正文>"}
