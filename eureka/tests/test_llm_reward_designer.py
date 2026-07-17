"""Unit tests for generate_candidates with mocked Groq key manager."""

from unittest.mock import MagicMock, patch

from eureka.llm_reward_designer import SYSTEM_PROMPT, _extract_code, generate_candidates

_VALID = "def shaping_reward(ego, road, info):\n    return 0.0\n"


def test_system_prompt_mentions_named_temperature_variable():
    lowered = SYSTEM_PROMPT.lower()
    assert "named" in lowered
    assert "temp" in lowered or "scale" in lowered


def test_system_prompt_forbids_nested_function_definitions():
    """Regression test: gen0 previously lost ~44% of candidates to
    'nested function definitions are not allowed' AST rejections. The
    system prompt must explicitly warn against this pattern."""
    lowered = SYSTEM_PROMPT.lower()
    assert "nested" in lowered or "inner" in lowered
    assert "def" in lowered


def test_system_prompt_forbids_lambda_expressions():
    """Regression test: gen0 was observed losing 3/8 candidates in one run
    to 'disallowed syntax: Lambda' smoke-test rejections (sandbox.py's AST
    allowlist deliberately excludes ast.Lambda - lambdas are a common
    dunder-access escape vector, e.g. the 'lambda_nested' red-team payload
    in test_redteam_sandbox.py). The system prompt must explicitly warn the
    model against reaching for lambda as an inline-function substitute."""
    lowered = SYSTEM_PROMPT.lower()
    assert "lambda" in lowered


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