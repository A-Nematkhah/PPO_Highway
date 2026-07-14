"""
sandbox.py

Shared AST whitelist + restricted exec loader for LLM-generated reward code.

Used by smoke_test.py (pre-train validation) and env_factory.py /
evaluate_candidate.py (training-time load). Replaces importlib.import_module
so candidate code never runs with the worker's full builtins.

`shaping_reward` may return either a bare float/int, or a 2-tuple
`(total_reward: float, reward_components: dict[str, float])` for component-
level reflection (EUREKA paper Prompt 3). Tuple/Dict/Return are already in
the AST allowlist, so no extra node types are required for the 2-tuple form.

Security model (Phase 1 hardening):
    - AST whitelist: only explicitly permitted node types / call targets pass.
      New Python syntax features default to REJECT until allowlisted.
    - Restricted exec namespace: same tiny builtin set + injected `math`.
    - Training-time load re-validates AST from disk (tamper detection).

Remaining risk (documented, Phase 3 target):
    - In-process exec() is not a cryptographic sandbox; a determined attacker
      may find escape vectors we did not anticipate.
    - No OS-level container (nsjail/firejail) yet — worker still shares the
      parent user's filesystem/network UID.
    - Declarative Reward DSL (no exec) is the long-term zero-exec path.
"""

from __future__ import annotations

import ast
import math
import os
from typing import Callable

# deliberately tiny whitelist - blocks import, open, exec, eval, __import__,
# and anything else that could do something unexpected. Includes common
# type names because the LLM prompt asks for `info: dict` annotations.
SAFE_BUILTINS: dict = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range,
    "float": float, "int": int, "bool": bool, "sum": sum, "sorted": sorted,
    "dict": dict, "list": list, "tuple": tuple, "str": str, "set": set,
    "isinstance": isinstance, "enumerate": enumerate, "zip": zip,
    "round": round, "pow": pow, "True": True, "False": False, "None": None,
}

_FORBIDDEN_CALL_NAMES = frozenset({
    "__import__", "eval", "exec", "open", "compile", "getattr", "setattr",
    "delattr", "hasattr", "type", "vars", "dir", "super", "locals", "globals",
    "input", "breakpoint", "help", "repr", "id", "memoryview", "bytes",
    "bytearray", "chr", "ord", "object", "classmethod", "staticmethod",
    "property", "iter", "next", "aiter", "anext",
})

_MATH_CALLABLES = frozenset(
    name for name in dir(math) if not name.startswith("_")
)

# Whitelist of AST node types permitted in candidate reward functions.
# Anything else (Import, Lambda, ClassDef, With, Try, ...) is rejected by default.
_ALLOWED_NODE_TYPES = frozenset({
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Expr,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.MatMult,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
    ast.And, ast.Or, ast.Not, ast.UAdd, ast.USub, ast.Invert,
})


class _WhitelistASTVisitor(ast.NodeVisitor):
    """
    Allowlist gate: reject any AST construct not explicitly permitted.
    Safer than a blacklist because forgotten new syntax defaults to blocked.
    """

    def __init__(self):
        self.error: str | None = None
        self._function_depth = 0

    def _reject(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def visit(self, node):
        if self.error is not None:
            return
        node_type = type(node)
        if node_type not in _ALLOWED_NODE_TYPES:
            self._reject(f"disallowed syntax: {node_type.__name__}")
            return
        super().visit(node)

    def visit_Module(self, node):
        # shaping_reward may return float OR (total, components: dict);
        # ast.Tuple / ast.Dict / ast.Return are already allowlisted above.
        fn_defs = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(fn_defs) != 1 or fn_defs[0].name != "shaping_reward":
            self._reject("module must define exactly one function named `shaping_reward`")
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._function_depth += 1
        if self._function_depth > 1:
            self._reject("nested function definitions are not allowed")
            self._function_depth -= 1
            return
        if node.name != "shaping_reward":
            self._reject(f"only `shaping_reward` is allowed, got {node.name!r}")
            self._function_depth -= 1
            return
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self._reject(f"dunder attribute access is not allowed: {node.attr!r}")
            return
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALL_NAMES:
                self._reject(f"call to {node.func.id!r} is not allowed")
                return
            if node.func.id not in SAFE_BUILTINS:
                self._reject(f"call to undefined builtin {node.func.id!r} is not allowed")
                return
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_CALL_NAMES:
                self._reject(f"call to {node.func.attr!r} is not allowed")
                return
            if node.func.attr == "format":
                self._reject(
                    "'.format()' calls are not allowed "
                    "(attribute traversal can bypass dunder checks)"
                )
                return
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr not in _MATH_CALLABLES
            ):
                self._reject(f"call to math.{node.func.attr!r} is not allowed")
                return
        self.generic_visit(node)


def normalize_shaping_output(raw_value) -> tuple[float, dict]:
    """
    Accepts whatever shaping_reward() returned and normalizes it to
    (total: float, components: dict[str, float]).

    - If raw_value is a plain int/float: returns (float(raw_value), {}).
    - If raw_value is a 2-tuple/list (total, components):
        - total must be int/float -> cast to float
        - components must be a dict; any non-numeric or non-finite
          values inside components are dropped (not errored) so a
          slightly malformed components dict doesn't crash training,
          consistent with this project's existing "degrade to zero
          shaping" philosophy in candidate_wrapper.py
    - Anything else (wrong arity tuple, wrong types, non-finite total)
      raises ValueError with a descriptive message. The CALLER (not
      this function) is responsible for catching that and degrading to
      0.0, mirroring how shaping_call.py already handles exceptions.
    """
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        if not math.isfinite(raw_value):
            raise ValueError(f"non-finite shaping total: {raw_value!r}")
        return float(raw_value), {}

    if isinstance(raw_value, (tuple, list)):
        if len(raw_value) != 2:
            raise ValueError(
                f"shaping_reward tuple must have length 2, got {len(raw_value)}"
            )
        total, components = raw_value
        if not isinstance(total, (int, float)) or isinstance(total, bool):
            raise ValueError(f"shaping total must be numeric, got {type(total).__name__}")
        if not math.isfinite(total):
            raise ValueError(f"non-finite shaping total: {total!r}")
        if not isinstance(components, dict):
            raise ValueError(
                f"reward_components must be a dict, got {type(components).__name__}"
            )
        cleaned: dict[str, float] = {}
        for key, value in components.items():
            if not isinstance(key, str):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if not math.isfinite(value):
                continue
            cleaned[key] = float(value)
        return float(total), cleaned

    raise ValueError(
        f"shaping_reward must return float or (float, dict), got {type(raw_value).__name__}"
    )


def validate_candidate_ast(code_str: str) -> tuple[bool, str]:
    """
    Parse `code_str` and reject anything outside the AST allowlist.
    Returns (passed, message).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    visitor = _WhitelistASTVisitor()
    visitor.visit(tree)
    if visitor.error is not None:
        return False, visitor.error
    return True, "ok"


def exec_shaping_reward(code_str: str) -> tuple[Callable | None, str | None]:
    """
    Validate and exec candidate code in a restricted namespace.
    Returns (shaping_reward_callable_or_None, error_message_or_None).
    """
    passed, message = validate_candidate_ast(code_str)
    if not passed:
        return None, message

    namespace: dict = {}
    try:
        exec(compile(code_str, "<candidate>", "exec"),
             {"__builtins__": SAFE_BUILTINS, "math": math}, namespace)
    except Exception as e:
        return None, f"exec failed: {e}"

    fn = namespace.get("shaping_reward")
    if fn is None or not callable(fn):
        return None, "no `shaping_reward` function defined"
    return fn, None


def module_path_to_source_path(module_path: str) -> str:
    """Map dotted module path to on-disk .py path relative to project root."""
    return os.path.join(*module_path.split(".")) + ".py"


def load_shaping_reward_from_code(code_str: str) -> Callable:
    """
    Load shaping_reward from a code string. Raises ValueError on validation/exec failure.
    """
    fn, error = exec_shaping_reward(code_str)
    if error is not None:
        raise ValueError(error)
    return fn


def load_shaping_reward_from_module_path(module_path: str) -> Callable:
    """
    Read candidate source from disk, re-validate AST, exec in restricted namespace.
    Raises FileNotFoundError / ValueError on failure.
    """
    source_path = module_path_to_source_path(module_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"candidate source not found: {source_path}")

    with open(source_path, encoding="utf-8") as f:
        code_str = f.read()

    return load_shaping_reward_from_code(code_str)
