import argparse
from collections.abc import Mapping
from pathlib import Path

import yaml


DEFAULT_ITSC_CONFIG = "results/rag_moe/experts_itsc_full_SH.yaml"
DEFAULT_RAFT_CONFIG = "results/rag_moe/itsc_raft_pipeline/experts_raft_alpha.yaml"
DEFAULT_OUTPUT = "results/rag_moe/itsc_raft_pipeline/experts_itsc_raft.yaml"


def _load_yaml(path):
    path = Path(path)
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"{path} YAML root must be a mapping")
    return config


def _expert_config(config, expert_name, path):
    experts = config.get("experts")
    if not isinstance(experts, Mapping):
        raise ValueError(f"{path} must define an experts mapping")
    expert = experts.get(expert_name)
    if not isinstance(expert, Mapping):
        raise ValueError(f"{path} must define experts.{expert_name} as a mapping")
    return expert


def write_pipeline_config(itsc_config_path, raft_config_path, output_path):
    itsc_config_path = Path(itsc_config_path)
    raft_config_path = Path(raft_config_path)
    output_path = Path(output_path)

    itsc_config = _load_yaml(itsc_config_path)
    raft_config = _load_yaml(raft_config_path)

    itsc_expert = dict(_expert_config(itsc_config, "itsc", itsc_config_path))
    raft_expert = dict(_expert_config(raft_config, "raft", raft_config_path))
    itsc_expert["mode"] = "residual"
    raft_expert["mode"] = "residual"

    output_config = {
        "experts": {
            "itsc": itsc_expert,
            "raft": raft_expert,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(output_config, handle, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a focused ITSC+RAFT expert pipeline config."
    )
    parser.add_argument("--itsc_config", default=DEFAULT_ITSC_CONFIG)
    parser.add_argument("--raft_config", default=DEFAULT_RAFT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    write_pipeline_config(args.itsc_config, args.raft_config, args.output)


if __name__ == "__main__":
    main()
