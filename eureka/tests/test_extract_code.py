"""Unit tests for eureka/llm_reward_designer.py::_extract_code."""

import pytest

from eureka.llm_reward_designer import _extract_code

_VALID = "def shaping_reward(ego, road, info):\n    return 0.0\n"


@pytest.mark.parametrize(
    "text,expected",
    [
        (f"```python\n{_VALID}```", _VALID.strip()),
        (f"Here is code:\n```python\n{_VALID}```\nThanks!", _VALID.strip()),
        (f"```\n{_VALID}```", _VALID.strip()),
        (f"Some prose\n{_VALID}", _VALID.strip()),
        ("no function here", None),
        ("", None),
        ("```python\nimport os\n```", "import os"),
    ],
)
def test_extract_code_paths(text, expected):
    result = _extract_code(text)
    if expected is None:
        assert result is None
    else:
        assert result == expected
