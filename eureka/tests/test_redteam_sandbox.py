"""Red-team sandbox escape payloads — all must be rejected by validate_candidate_ast."""

import pytest

from eureka.sandbox import validate_candidate_ast

# Each entry: (name, code snippet wrapping malicious body in shaping_reward)
_REDTEAM_PAYLOADS = [
    (
        "import_os",
        "import os\n\ndef shaping_reward(ego, road, info):\n    return 0.0\n",
    ),
    (
        "import_from",
        "from os import system\n\ndef shaping_reward(ego, road, info):\n    return 0.0\n",
    ),
    (
        "dunder_mro",
        "def shaping_reward(ego, road, info):\n"
        "    return ().__class__.__bases__[0].__subclasses__()\n",
    ),
    (
        "format_bypass",
        'def shaping_reward(ego, road, info):\n'
        '    return "{0.__class__.__bases__}".format(ego)\n',
    ),
    (
        "fstring_dunder",
        'def shaping_reward(ego, road, info):\n'
        '    return f"{ego.__class__}"\n',
    ),
    (
        "eval_call",
        "def shaping_reward(ego, road, info):\n    return eval('1')\n",
    ),
    (
        "exec_call",
        "def shaping_reward(ego, road, info):\n    exec('pass')\n    return 0.0\n",
    ),
    (
        "open_call",
        "def shaping_reward(ego, road, info):\n    open('/etc/passwd')\n    return 0.0\n",
    ),
    (
        "getattr_dunder",
        "def shaping_reward(ego, road, info):\n"
        "    return getattr(ego, '__class__')\n",
    ),
    (
        "lambda_nested",
        "def shaping_reward(ego, road, info):\n"
        "    f = lambda: __import__('os')\n    return 0.0\n",
    ),
    (
        "class_def",
        "class Evil:\n    pass\n\ndef shaping_reward(ego, road, info):\n    return 0.0\n",
    ),
    (
        "try_except_import",
        "def shaping_reward(ego, road, info):\n"
        "    try:\n        import os\n    except Exception:\n        pass\n"
        "    return 0.0\n",
    ),
    (
        "compile_call",
        "def shaping_reward(ego, road, info):\n"
        "    compile('pass', '<x>', 'exec')\n    return 0.0\n",
    ),
    (
        "globals_call",
        "def shaping_reward(ego, road, info):\n    return globals()\n",
    ),
    (
        "nested_function",
        "def shaping_reward(ego, road, info):\n"
        "    def inner():\n        return 1\n    return inner()\n",
    ),
    (
        "ego_attribute_mutation",
        "def shaping_reward(ego, road, info):\n"
        "    ego.speed = 999.0\n"
        "    return 0.0\n",
    ),
    (
        "ego_attribute_mutation_nested",
        "def shaping_reward(ego, road, info):\n"
        "    ego.position[0] = 0.0\n"
        "    return 0.0\n",
    ),
    (
        "road_subscript_mutation",
        "def shaping_reward(ego, road, info):\n"
        "    road.vehicles[0] = None\n"
        "    return 0.0\n",
    ),
    (
        "road_attribute_delete",
        "def shaping_reward(ego, road, info):\n"
        "    del road.vehicles\n"
        "    return 0.0\n",
    ),
    (
        "info_subscript_mutation",
        "def shaping_reward(ego, road, info):\n"
        "    info['crashed'] = False\n"
        "    return 0.0\n",
    ),
]


@pytest.mark.parametrize("name,code", _REDTEAM_PAYLOADS, ids=[p[0] for p in _REDTEAM_PAYLOADS])
def test_redteam_payload_rejected(name, code):
    passed, message = validate_candidate_ast(code)
    assert passed is False, f"{name} unexpectedly passed: {message}"
    assert message
