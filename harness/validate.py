"""Static checks on agent selector files, and loading them.

A selector may import only torch / math / numpy / functools / typing / dataclasses and
harness.selector_api. File access, process control, dynamic code and any teacher/label/test access
are rejected before the job is queued, and again when the job starts.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

from . import config as C
from .baselines import BUILTIN

ALLOWED_IMPORTS = {"torch", "math", "numpy", "functools", "typing", "dataclasses", "harness.selector_api",
                   "__future__"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "input", "breakpoint", "memoryview"}
FORBIDDEN_ATTRS = {"load", "save", "hub", "backward", "teacher", "targets", "labels", "test", "manual_seed",
                   "set_rng_state", "__dict__", "__class__", "__subclasses__", "__globals__", "__code__",
                   "system", "popen", "environ", "from_file", "fromfile", "loadtxt", "genfromtxt", "memmap",
                   "sys", "os", "builtins", "subprocess", "importlib", "_C", "ctypes", "ctypeslib", "inspect",
                   "gc", "modules", "pickle", "serialization", "multiprocessing", "distributed",
                   "cpp_extension", "f2py", "socket", "urllib", "request"}
META_KEYS = ("method", "paper", "why", "vs_maskedkd", "prior_art")
SELECTORS_DIR = C.ROOT / "selectors"


class SelectorRejected(Exception):
    pass


def check_source(src: str, name: str):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SelectorRejected(f"{name}: syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS and a.name not in ALLOWED_IMPORTS:
                    raise SelectorRejected(f"{name}: import '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level != 0 or (mod not in ALLOWED_IMPORTS and mod.split(".")[0] not in ALLOWED_IMPORTS - {"harness"}):
                raise SelectorRejected(f"{name}: 'from {mod} import' not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SelectorRejected(f"{name}: name '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise SelectorRejected(f"{name}: attribute '.{node.attr}' not allowed")
    meta = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "META" for t in node.targets):
            try:
                meta = ast.literal_eval(node.value)
            except ValueError:
                raise SelectorRejected(f"{name}: META must be a literal dict")
    if not isinstance(meta, dict):
        raise SelectorRejected(f"{name}: missing META dict")
    for k in META_KEYS:
        if not isinstance(meta.get(k), str) or len(meta[k].strip()) < 3:
            raise SelectorRejected(f"{name}: META['{k}'] must be a non-empty string")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Selector" for n in tree.body):
        raise SelectorRejected(f"{name}: missing class Selector")
    return meta


def selector_source(name: str):
    if name in BUILTIN:
        return None
    path = SELECTORS_DIR / f"{name}.py"
    if not path.exists():
        raise SelectorRejected(f"no selector file {path}")
    return path.read_text()


def load_selector(name: str, src: str = None, kwargs=None):
    """Returns (instance, meta, sha256 of source). For builtins src/meta are None."""
    kwargs = kwargs or {}
    if name in BUILTIN:
        return BUILTIN[name](**kwargs), {"method": f"builtin:{name}"}, "builtin"
    src = src if src is not None else selector_source(name)
    meta = check_source(src, name)
    spec = importlib.util.spec_from_loader(f"selectors_{name}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"selectors/{name}.py", "exec"), mod.__dict__)
    return mod.Selector(**kwargs), meta, hashlib.sha256(src.encode()).hexdigest()
