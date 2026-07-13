"""Unit tests for generate_candidates with mocked Groq key manager."""

from unittest.mock import MagicMock, patch

from eureka.llm_reward_designer import SYSTEM_PROMPT, _extract_code, generate_candidates

_VALID = "def shaping_reward(ego, road, info):\n    return 0.0\n"


def test_system_prompt_mentions_named_temperature_variable():
    lowered = SYSTEM_PROMPT.lower()
    assert "named" in lowered
    assert "temp" in lowered or "scale" in lowered


@patch("key_manager.get_key_manager")
def test_generate_candidates_parses_fenced_response(mock_get_km):
    manager = MagicMock()
    mock_get_km.return_value = manager
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=f"```python\n{_VALID}```"))]
    manager.chat_completion.return_value = response

    codes = generate_candidates(None, k=2, generation=0, model="test-model", temperature=0.5)
    assert len(codes) == 2
    assert _extract_code(codes[0]) == _VALID.strip()


@patch("key_manager.get_key_manager")
def test_generate_candidates_skips_unparseable_response(mock_get_km):
    manager = MagicMock()
    mock_get_km.return_value = manager
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="no code here"))]
    manager.chat_completion.return_value = response

    codes = generate_candidates(None, k=1, generation=0, model="test-model", temperature=0.5)
    assert codes == []
