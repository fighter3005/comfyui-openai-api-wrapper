import os
import importlib.util
from typing import Optional

_MODULE_CACHE = {}

def _module_path_for_model(model_id: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{model_id}.py")

def load_model_module(model_id: str):
    """
    Loads a workflow module from workflows/{model_id}.py.
    This supports hyphenated filenames (e.g. qwen-image.py) by loading from file path.
    """
    path = _module_path_for_model(model_id)
    if not os.path.exists(path):
        return None

    if path in _MODULE_CACHE:
        return _MODULE_CACHE[path]

    module_name = f"workflow_{model_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULE_CACHE[path] = mod
    return mod

def get_supported_models():
    """
    Returns model ids based on filenames inside workflows/ (excluding __init__.py).
    """
    out = []
    for fn in os.listdir(os.path.dirname(__file__)):
        if not fn.endswith(".py"):
            continue
        if fn == "__init__.py":
            continue
        out.append(fn[:-3])  # strip .py
    return sorted(out)

def get_workflow(model_id: str, **kwargs) -> Optional[dict]:
    """
    Calls workflows/{model_id}.py:get_workflow(**kwargs) and returns a workflow dict.
    """
    mod = load_model_module(model_id)
    if not mod or not hasattr(mod, "get_workflow"):
        return None
    return mod.get_workflow(**kwargs)