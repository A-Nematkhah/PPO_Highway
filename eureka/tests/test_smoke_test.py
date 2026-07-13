"""Unit tests for eureka/smoke_test.py AST gate and rejection behavior."""

import pytest

from eureka.sandbox import validate_candidate_ast
from eureka.smoke_test import smoke_test

_VALID_CANDIDATE = """
def shaping_reward(ego, road, info):
    return float(info.get("n_overtakes", 0)) * 0.1
"""


@pytest.mark.parametrize(
    "code,expected_substr",
    [
        ("import os\n" + _VALID_CANDIDATE, "import"),
        ("from math import sqrt\n" + _VALID_CANDIDATE, "import"),
        (
            "def shaping_reward(ego, road, info):\n"
            "    return ().__class__.__bases__[0].__subclasses__()\n",
            "dunder",
        ),
        (
            "def shaping_reward(ego, road, info):\n"
            "    return eval('1')\n",
            "eval",
        ),
        (
            "def shaping_reward(ego, road, info):\n"
            "    open('x', 'w')\n"
            "    return 0.0\n",
            "open",
        ),
        (
            "def shaping_reward(ego, road, info):\n"
            "    global x\n"
            "    return 0.0\n",
            "global",
        ),
        (
            "def shaping_reward(ego, road, info):\n"
            '    return "{0.__class__.__bases__}".format(ego)\n',
            "format",
        ),
        (
            'def shaping_reward(ego, road, info):\n'
            '    return "{}".format(1)\n',
            "format",
        ),
        (
            "def shaping_reward(ego, road, info):\n"
            "    return f\"{ego.__class__}\"\n",
            "joinedstr",
        ),
    ],
)
def test_validate_candidate_ast_rejects_forbidden_constructs(code, expected_substr):
    passed, message = validate_candidate_ast(code)
    assert passed is False
    assert expected_substr in message.lower()


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + _VALID_CANDIDATE,
        (
            "def shaping_reward(ego, road, info):\n"
            "    return ().__class__.__bases__[0].__subclasses__()\n"
        ),
    ],
)
def test_smoke_test_rejects_malicious_code_without_passing(code):
    """smoke_test must reject the exact string that would be saved to disk."""
    passed, message = smoke_test(code, n_trials=1)
    assert passed is False
    assert message  # non-empty rejection reason


def test_validate_candidate_ast_accepts_minimal_candidate():
    passed, message = validate_candidate_ast(_VALID_CANDIDATE)
    assert passed is True
    assert message == "ok"


def test_validate_candidate_ast_accepts_generator_expression():
    code = (
        "def shaping_reward(ego, road, info):\n"
        "    return sum(v.speed for v in road.vehicles if v is not ego) * 0.01\n"
    )
    passed, message = validate_candidate_ast(code)
    assert passed is True
    assert message == "ok"


def test_smoke_test_accepts_valid_candidate():
    passed, message = smoke_test(_VALID_CANDIDATE, n_trials=2)
    assert passed is True, message


def test_smoke_test_accepts_component_tuple_return():
    code = (
        "def shaping_reward(ego, road, info):\n"
        "    return 0.1, {'a': 0.05, 'b': 0.05}\n"
    )
    passed, message = smoke_test(code, n_trials=2)
    assert passed is True, message


def test_smoke_test_rejects_malformed_tuple_return():
    code = (
        "def shaping_reward(ego, road, info):\n"
        "    return 0.1, 0.2, 0.3\n"
    )
    passed, message = smoke_test(code, n_trials=1)
    assert passed is False
    assert "length 2" in message.lower() or "tuple" in message.lower()
