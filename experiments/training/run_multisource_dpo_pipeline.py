import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.training.protocol import DEFAULT_CITIES, parse_city_list, split_source_target


STAGE_NAMES = [
    "backbone",
    "experts",
    "candidate_cache",
    "supervised_router",
    "dpo_router",
    "zero_shot_eval",
]


def build_pipeline_plan(cities, target_city, run_root="results/training", run_tag="multisource_dpo", skip_dpo=False, expert_config=""):
    city_list = parse_city_list(cities)
    source_cities, target = split_source_target(city_list, target_city)
    run_dir = Path(run_root) / run_tag
    stage_dirs = {name: run_dir / name for name in STAGE_NAMES}
    backbone_ckpt = stage_dirs["backbone"] / "best_backbone.pt"
    expert_config_path = Path(expert_config) if expert_config else stage_dirs["experts"] / "experts_multisource.yaml"
    candidate_cache = stage_dirs["candidate_cache"] / "source_candidates.pt"
    supervised_router = stage_dirs["supervised_router"] / "best_router.pt"
    dpo_router = stage_dirs["dpo_router"] / "best_router.pt"
    final_router = supervised_router if skip_dpo else dpo_router

    stages = [
        {
            "name": "backbone",
            "uses_cities": list(source_cities),
            "output": str(backbone_ckpt),
            "inputs": {},
        },
        {
            "name": "experts",
            "uses_cities": list(source_cities),
            "output": str(expert_config_path),
            "inputs": {"backbone_ckpt": str(backbone_ckpt)},
        },
        {
            "name": "candidate_cache",
            "uses_cities": list(source_cities),
            "output": str(candidate_cache),
            "inputs": {
                "backbone_ckpt": str(backbone_ckpt),
                "expert_config": str(expert_config_path),
            },
        },
        {
            "name": "supervised_router",
            "uses_cities": list(source_cities),
            "output": str(supervised_router),
            "inputs": {"candidate_cache": str(candidate_cache)},
        },
    ]
    if not skip_dpo:
        stages.append(
            {
                "name": "dpo_router",
                "uses_cities": list(source_cities),
                "output": str(dpo_router),
                "inputs": {
                    "candidate_cache": str(candidate_cache),
                    "router_ckpt": str(supervised_router),
                },
            }
        )
    stages.append(
        {
            "name": "zero_shot_eval",
            "uses_cities": [target],
            "output": str(stage_dirs["zero_shot_eval"] / "summary.json"),
            "inputs": {
                "backbone_ckpt": str(backbone_ckpt),
                "expert_config": str(expert_config_path),
                "router_ckpt": str(final_router),
            },
        }
    )
    return {
        "run_dir": str(run_dir),
        "cities": city_list,
        "source_cities": source_cities,
        "target_city": target,
        "stages": stages,
    }


def write_dry_run_plan(plan):
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "pipeline_plan.json"
    text_path = run_dir / "pipeline_plan.txt"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "Multi-source zero-shot RAG-MoE training plan",
        "target_city: %s" % plan["target_city"],
        "source_cities: %s" % ",".join(plan["source_cities"]),
        "run_dir: %s" % plan["run_dir"],
        "",
        "stages:",
    ]
    for stage in plan["stages"]:
        lines.append(
            "- {name}: cities={cities} output={output}".format(
                name=stage["name"],
                cities=",".join(stage["uses_cities"]),
                output=stage["output"],
            )
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return text_path


def _status_path(plan):
    return Path(plan["run_dir"]) / "pipeline_status.json"


def _write_status(plan, status):
    path = _status_path(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _stage_by_name(plan):
    return {stage["name"]: stage for stage in plan["stages"]}


def run_pipeline(args, stage_runner=None):
    plan = build_pipeline_plan(
        args.cities,
        args.target_city,
        args.run_root,
        args.run_tag,
        skip_dpo=getattr(args, "skip_dpo", False),
        expert_config=getattr(args, "expert_config", ""),
    )
    write_dry_run_plan(plan)
    if getattr(args, "skip_experts", False) and not getattr(args, "expert_config", ""):
        raise ValueError("--skip_experts requires --expert_config")

    status = {
        stage["name"]: {
            "status": "pending",
            "output": stage["output"],
            "inputs": stage.get("inputs", {}),
        }
        for stage in plan["stages"]
    }
    _write_status(plan, status)
    runner = stage_runner or run_stage
    for stage in plan["stages"]:
        name = stage["name"]
        if _should_skip_stage(name, args):
            status[name]["status"] = "skipped"
            _write_status(plan, status)
            continue
        output_path = Path(stage["output"])
        if getattr(args, "resume", False) and output_path.exists():
            status[name]["status"] = "skipped_existing"
            _write_status(plan, status)
            continue
        status[name]["status"] = "running"
        _write_status(plan, status)
        try:
            result = runner(stage, args, plan)
            if result is not None:
                status[name]["result"] = result
            if not output_path.exists():
                raise RuntimeError("stage %s did not produce expected output %s" % (name, output_path))
        except Exception as exc:
            status[name]["status"] = "failed"
            status[name]["error"] = str(exc)
            _write_status(plan, status)
            raise
        status[name]["status"] = "completed"
        _write_status(plan, status)
    return {"plan": plan, "status": status}


def _should_skip_stage(stage_name, args):
    if stage_name == "backbone" and getattr(args, "skip_backbone", False):
        return True
    if stage_name == "experts" and getattr(args, "skip_experts", False):
        return True
    if stage_name == "dpo_router" and getattr(args, "skip_dpo", False):
        return True
    return False


def _common_kwargs(args):
    return {
        "cities": args.cities,
        "target_city": args.target_city,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "base_lr": args.base_lr,
        "patience": args.patience,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "node_dim": args.node_dim,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "embed_dim": args.embed_dim,
        "num_layer": args.num_layer,
        "mp_layers": args.mp_layers,
        "llm_enc_dim": args.llm_enc_dim,
        "num_unknown_nodes": args.num_unknown_nodes,
        "num_masked_nodes": args.num_masked_nodes,
        "device": args.device,
    }


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def run_stage(stage, args, plan):
    name = stage["name"]
    if name == "backbone":
        from experiments.training.train_backbone_multisource import train_backbone_multisource

        return train_backbone_multisource(
            _namespace(
                **_common_kwargs(args),
                run_dir=str(Path(plan["run_dir"]) / "backbone"),
                max_grad_norm=args.max_grad_norm,
            )
        )
    if name == "experts":
        from experiments.training.train_experts_multisource import train_experts_multisource_from_contexts, _load_backbone
        from experiments.training.multisource_data import build_city_contexts

        expert_args = _namespace(
            **_common_kwargs(args),
            run_dir=str(Path(plan["run_dir"]) / "experts"),
            backbone_ckpt=stage["inputs"]["backbone_ckpt"],
            enabled_experts=args.enabled_experts,
            residual_l1=args.residual_l1,
            early_stop_min_delta=args.early_stop_min_delta,
            calibration_scope=args.calibration_scope,
            cal_lr=args.cal_lr,
            identity_l1=args.identity_l1,
            source_top_k=args.source_top_k,
            source_temperature=args.source_temperature,
            source_confidence_threshold=args.source_confidence_threshold,
            source_alpha_lr=args.source_alpha_lr,
            source_alpha_mode=args.source_alpha_mode,
            min_top1=args.min_top1,
            min_margin=args.min_margin,
            max_entropy=args.max_entropy,
            score_chunk_size=args.score_chunk_size,
            volatility_hidden_dim=args.volatility_hidden_dim,
            max_abs_delta=args.max_abs_delta,
            identity_l2=args.identity_l2,
            max_delta_penalty=args.max_delta_penalty,
            min_history_std=args.min_history_std,
            min_history_max=args.min_history_max,
            itsc_top_k=args.itsc_top_k,
            itsc_temperature=args.itsc_temperature,
            raft_top_k=args.raft_top_k,
            raft_temperature=args.raft_temperature,
            raft_alpha_lr=args.raft_alpha_lr,
            raft_alpha_mode=args.raft_alpha_mode,
            tpb_top_k=args.tpb_top_k,
            tpb_temperature=args.tpb_temperature,
            tpb_patch_len=args.tpb_patch_len,
            tpb_max_patterns=args.tpb_max_patterns,
            itsc_gate_lr=args.itsc_gate_lr,
            itsc_gate_l1=args.itsc_gate_l1,
            itsc_gate_hidden_dim=args.itsc_gate_hidden_dim,
            min_gate_available=args.min_gate_available,
        )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        protocol, contexts = build_city_contexts(
            args.cities.split(","),
            args.target_city,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_unknown_nodes=args.num_unknown_nodes,
            num_masked_nodes=args.num_masked_nodes,
            seed=args.seed,
        )
        backbone = _load_backbone(expert_args, device)
        return train_experts_multisource_from_contexts(expert_args, backbone, contexts, device, protocol.source_cities, protocol.target_city)
    if name == "candidate_cache":
        from experiments.training.cache_candidates import build_rag_moe_for_cache, collect_candidate_cache, save_candidate_cache
        from experiments.training.multisource_data import build_city_contexts

        cache_args = _namespace(
            **_common_kwargs(args),
            run_dir=str(Path(plan["run_dir"]) / "candidate_cache"),
            backbone_ckpt=stage["inputs"]["backbone_ckpt"],
            enabled_experts=args.enabled_experts,
            expert_config=stage["inputs"]["expert_config"],
            router_config=args.router_config,
        )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        protocol, contexts = build_city_contexts(
            args.cities.split(","),
            args.target_city,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_unknown_nodes=args.num_unknown_nodes,
            num_masked_nodes=args.num_masked_nodes,
            seed=args.seed,
        )
        model = build_rag_moe_for_cache(cache_args, protocol.source_cities, device)
        payloads = [collect_candidate_cache(model, contexts, split=split, device=device) for split in ("train", "val")]
        payload = dict(payloads[0])
        for key in ("features", "available", "candidates", "labels", "candidate_errors"):
            payload[key] = torch.cat([item[key] for item in payloads], dim=0)
        metadata = {key: [] for key in payloads[0]["metadata"]}
        for item in payloads:
            for key, value in item["metadata"].items():
                metadata[key].extend(value)
        payload["metadata"] = metadata
        save_candidate_cache(stage["output"], payload)
        return {"num_samples": int(payload["features"].shape[0])}
    if name == "supervised_router":
        from experiments.training.train_router_supervised import train_router_supervised_from_cache

        return train_router_supervised_from_cache(
            cache_path=stage["inputs"]["candidate_cache"],
            run_dir=str(Path(plan["run_dir"]) / "supervised_router"),
            hidden_dim=args.router_hidden_dim,
            max_epochs=args.router_max_epochs,
            lr=args.router_lr,
            oracle_margin=args.oracle_margin,
            lambda_sparse=args.lambda_sparse,
            patience=args.router_patience,
            dropout=args.router_dropout,
            device=args.device or None,
        )
    if name == "dpo_router":
        from experiments.training.train_router_dpo import train_router_dpo_from_cache

        return train_router_dpo_from_cache(
            cache_path=stage["inputs"]["candidate_cache"],
            router_ckpt=stage["inputs"]["router_ckpt"],
            run_dir=str(Path(plan["run_dir"]) / "dpo_router"),
            hidden_dim=args.router_hidden_dim,
            beta=args.dpo_beta,
            rel_margin=args.dpo_rel_margin,
            abs_margin=args.dpo_abs_margin,
            max_epochs=args.dpo_max_epochs,
            lr=args.dpo_lr,
            train_scope=args.train_scope,
            device=args.device or None,
        )
    if name == "zero_shot_eval":
        from experiments.training.cache_candidates import build_rag_moe_for_cache
        from experiments.training.evaluate_zero_shot import evaluate_zero_shot_model
        from experiments.training.multisource_data import build_target_city_context

        source_cities, _ = split_source_target(args.cities, args.target_city)
        eval_args = _namespace(
            **_common_kwargs(args),
            run_dir=str(Path(plan["run_dir"]) / "zero_shot_eval"),
            backbone_ckpt=stage["inputs"]["backbone_ckpt"],
            router_ckpt=stage["inputs"]["router_ckpt"],
            enabled_experts=args.enabled_experts,
            expert_config=stage["inputs"]["expert_config"],
            router_config=args.router_config,
        )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        context = build_target_city_context(
            args.cities.split(","),
            args.target_city,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_unknown_nodes=args.num_unknown_nodes,
            num_masked_nodes=args.num_masked_nodes,
            seed=args.seed,
        )
        model = build_rag_moe_for_cache(eval_args, source_cities, device)
        return evaluate_zero_shot_model(
            model=model,
            context=context,
            router_ckpt=stage["inputs"]["router_ckpt"],
            run_dir=str(Path(plan["run_dir"]) / "zero_shot_eval"),
            device=device,
            min_best_prob=args.min_best_prob,
            min_margin_over_none=args.min_margin_over_none,
        )
    raise ValueError("unknown stage %r" % name)


def build_parser():
    parser = argparse.ArgumentParser(description="Plan or run multi-source zero-shot RAG-MoE training.")
    parser.add_argument("--cities", type=str, default=",".join(DEFAULT_CITIES))
    parser.add_argument("--target_city", type=str, required=True)
    parser.add_argument("--run_root", type=str, default="results/training")
    parser.add_argument("--run_tag", type=str, default="multisource_dpo")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_backbone", action="store_true")
    parser.add_argument("--skip_experts", action="store_true")
    parser.add_argument("--skip_dpo", action="store_true")
    parser.add_argument("--expert_config", default="")
    parser.add_argument("--router_config", default="configs/rag_moe/router.yaml")
    parser.add_argument("--enabled_experts", default="all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--input_len", type=int, default=24)
    parser.add_argument("--output_len", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--mp_layers", type=int, default=1)
    parser.add_argument("--llm_enc_dim", type=int, default=4096)
    parser.add_argument("--num_unknown_nodes", type=int, default=10)
    parser.add_argument("--num_masked_nodes", type=int, default=6)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--residual_l1", type=float, default=0.0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--calibration_scope", choices=["shared", "horizon"], default="horizon")
    parser.add_argument("--cal_lr", type=float, default=0.01)
    parser.add_argument("--identity_l1", type=float, default=0.01)
    parser.add_argument("--source_top_k", type=int, default=1)
    parser.add_argument("--source_temperature", type=float, default=0.1)
    parser.add_argument("--source_confidence_threshold", type=float, default=0.5)
    parser.add_argument("--source_alpha_lr", type=float, default=0.01)
    parser.add_argument("--source_alpha_mode", choices=["scalar", "horizon"], default="scalar")
    parser.add_argument("--min_top1", type=float, default=0.7)
    parser.add_argument("--min_margin", type=float, default=0.02)
    parser.add_argument("--max_entropy", type=float, default=0.85)
    parser.add_argument("--score_chunk_size", type=int, default=2048)
    parser.add_argument("--volatility_hidden_dim", type=int, default=8)
    parser.add_argument("--max_abs_delta", type=float, default=0.5)
    parser.add_argument("--identity_l2", type=float, default=0.001)
    parser.add_argument("--max_delta_penalty", type=float, default=0.01)
    parser.add_argument("--min_history_std", type=float, default=0.0)
    parser.add_argument("--min_history_max", type=float, default=0.0)
    parser.add_argument("--itsc_top_k", type=int, default=5)
    parser.add_argument("--itsc_temperature", type=float, default=0.2)
    parser.add_argument("--raft_top_k", type=int, default=5)
    parser.add_argument("--raft_temperature", type=float, default=0.1)
    parser.add_argument("--raft_alpha_lr", type=float, default=0.01)
    parser.add_argument("--raft_alpha_mode", choices=["scalar", "horizon"], default="scalar")
    parser.add_argument("--tpb_top_k", type=int, default=8)
    parser.add_argument("--tpb_temperature", type=float, default=1.0)
    parser.add_argument("--tpb_patch_len", type=int, default=12)
    parser.add_argument("--tpb_max_patterns", type=int, default=20000)
    parser.add_argument("--itsc_gate_lr", type=float, default=0.001)
    parser.add_argument("--itsc_gate_l1", type=float, default=0.01)
    parser.add_argument("--itsc_gate_hidden_dim", type=int, default=8)
    parser.add_argument("--min_gate_available", type=float, default=0.0)
    parser.add_argument("--router_hidden_dim", type=int, default=128)
    parser.add_argument("--router_max_epochs", type=int, default=20)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--router_patience", type=int, default=10)
    parser.add_argument("--router_dropout", type=float, default=0.1)
    parser.add_argument("--oracle_margin", type=float, default=0.98)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--dpo_max_epochs", type=int, default=20)
    parser.add_argument("--dpo_lr", type=float, default=1e-3)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--dpo_rel_margin", type=float, default=0.01)
    parser.add_argument("--dpo_abs_margin", type=float, default=0.01)
    parser.add_argument("--train_scope", choices=["heads", "all"], default="heads")
    parser.add_argument("--min_best_prob", type=float, default=0.5)
    parser.add_argument("--min_margin_over_none", type=float, default=0.0)
    return parser


def main():
    args = build_parser().parse_args()
    plan = build_pipeline_plan(
        args.cities,
        args.target_city,
        args.run_root,
        args.run_tag,
        skip_dpo=args.skip_dpo,
        expert_config=args.expert_config,
    )
    path = write_dry_run_plan(plan)
    print("wrote pipeline plan: %s" % path)
    if args.dry_run:
        return
    result = run_pipeline(args)
    print(json.dumps(result["status"], sort_keys=True))


if __name__ == "__main__":
    main()
