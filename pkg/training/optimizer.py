
from torch.optim import Adam, AdamW

_OPTIMIZER_REGISTRY = {
    "Adam":Adam,
    "AdamW":AdamW,
}

def build_optimizer(cfg, model_params):

    if not "name" in cfg:
        raise KeyError(f"Optimizer name not specified")
    
    name = cfg["name"]
    if not name in _OPTIMIZER_REGISTRY:
        raise KeyError(f"Unknown optimizer {name}. Options: {list(_OPTIMIZER_REGISTRY)}")
    
    return _OPTIMIZER_REGISTRY[name](params=model_params,**cfg["params"])