from pathlib import Path

import yaml


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rag_moe_configs(expert_config_path, router_config_path):
    experts = load_yaml(expert_config_path)
    router = load_yaml(router_config_path)
    if "experts" not in experts:
        raise ValueError("expert config must contain an 'experts' mapping")
    if "router" not in router:
        raise ValueError("router config must contain a 'router' mapping")
    return experts, router
