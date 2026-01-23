"""Prompt tests."""

from src.prompts import PromptContext, default_registry, parse_json_output


def test_prompt_registry_lists_prompts() -> None:
    registry = default_registry()
    specs = registry.list()
    assert any(spec.id == "nsfw" for spec in specs)


def test_prompt_render_includes_labels() -> None:
    registry = default_registry()
    prompt = registry.get("quality")
    rendered = prompt.render(PromptContext(labels=["a", "b"]))
    assert "a, b" in rendered["user"]


def test_parse_json_output_handles_code_fence() -> None:
    text = "```json\n{\"label\": \"safe\", \"confidence\": 0.9}\n```"
    parsed = parse_json_output(text)
    assert parsed["label"] == "safe"
