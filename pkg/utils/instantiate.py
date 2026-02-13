import importlib

import importlib
from typing import Any, Mapping

def _load_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def instantiate_from_classpath(class_path, init_args):
    cls = _load_class(class_path)
    return cls(**init_args)

def instantiate(cfg: Any) -> Any:
    """
    Recursively instantiate objects from configs containing:
      - {"class_path": "...", "init_args": {...}}
    Also supports your earlier naming:
      - {"classpath": "...", "params": {...}}

    Recurses through dicts/lists/tuples and instantiates nested objects too.
    """
    # Pass through primitives
    if cfg is None or isinstance(cfg, (str, int, float, bool)):
        return cfg

    # Recurse lists/tuples
    if isinstance(cfg, list):
        return [instantiate(x) for x in cfg]
    if isinstance(cfg, tuple):
        return tuple(instantiate(x) for x in cfg)

    # Recurse dict-like
    if isinstance(cfg, Mapping):
        # Detect "instantiable" node
        if ("class" in cfg):
            class_path = cfg.get("class")
            init_args = cfg.get("args", {})

            # Recursively instantiate init args first
            init_args = {k: instantiate(v) for k, v in init_args.items()}

            cls = _load_class(class_path)
            return cls(**init_args)

        # Plain dict: just recurse into values
        return {k: instantiate(v) for k, v in cfg.items()}

    # Anything else: return as-is
    return cfg
