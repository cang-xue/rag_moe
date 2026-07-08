import os
from typing import Iterable

import torch


def require_config_keys(expert_name, config, keys: Iterable[str]):
    missing = [key for key in keys if not config.get(key)]
    if missing:
        raise ValueError(
            "%s full_model requires config keys: %s" % (
                expert_name,
                ", ".join(missing),
            )
        )


def require_artifact(expert_name, key, path):
    if not path:
        raise FileNotFoundError("%s %s is required but empty" % (expert_name, key))
    if not os.path.exists(path):
        raise FileNotFoundError("%s %s does not exist: %s" % (expert_name, key, path))
    return path


def load_torch_artifact(expert_name, key, path, map_location="cpu"):
    require_artifact(expert_name, key, path)
    try:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        raise RuntimeError(
            "%s failed to load %s=%s: %s" % (expert_name, key, path, exc)
        ) from exc


def extract_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state
