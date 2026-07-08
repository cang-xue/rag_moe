import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.training.multisource_data import build_city_contexts
from experiments.training.protocol import DEFAULT_CITIES, split_source_target
from src.rag_moe.experts.custom_features import history_node_features, horizon_fraction
from src.rag_moe.experts.itsc import ITSCExpert
from src.rag_moe.experts.itsc_segment_gate import ITSCSegmentGateExpert
from src.rag_moe.experts.raft import RAFTExpert
from src.rag_moe.experts.source_window import SourceWindowExpert
from src.rag_moe.experts.tpb import TPBExpert
from src.rag_moe.experts.volatility_peak import VolatilityPeakResidual
from src.utils.helper import move_batch_meta, split_batch


ALL_MULTISOURCE_EXPERTS = (
    "itsc",
    "raft",
    "calibration",
    "source_window",
    "volatility_peak",
    "itsc_segment_gate",
)
SUPPORTED_EXPERTS = ALL_MULTISOURCE_EXPERTS


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return _jsonable(tensor.item() if tensor.numel() == 1 else tensor.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def _write_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_backbone(args, device):
    from src.models.impel import IMPEL

    model = IMPEL(
        node_dim=args.node_dim,
        input_len=args.input_len,
        in_dim=args.input_dim,
        embed_dim=args.embed_dim,
        output_len=args.output_len,
        num_layer=args.num_layer,
        name="impel",
        dataset="multisource_to_%s" % args.target_city,
        device=device,
        num_nodes=0,
        seq_len=args.seq_len,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        llm_enc_dim=args.llm_enc_dim,
        supports_len=1,
        mp_layers=args.mp_layers,
    ).to(device)
    state = _extract_state_dict(_safe_torch_load(args.backbone_ckpt, map_location=device))
    model.load_state_dict(state, strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _inverse_transform(tensor, scalers):
    output = tensor.clone()
    for node_id, scaler in enumerate(scalers):
        output[:, :, node_id, :1] = scaler.inverse_transform(output[:, :, node_id, :1])
    return output


def _sample_ids_from_meta(batch_meta, batch_size):
    sample_ids = batch_meta.get("sample_ids")
    if sample_ids is None:
        return list(range(batch_size))
    if hasattr(sample_ids, "detach"):
        sample_ids = sample_ids.detach().cpu()
    if sample_ids.dim() > 1:
        sample_ids = sample_ids[:, 0]
    return [int(value) for value in sample_ids.reshape(-1).tolist()]


@torch.no_grad()
def collect_expert_payloads(backbone, contexts, device, output_len=24, output_dim=1):
    backbone = backbone.to(device)
    backbone.eval()
    payloads = {"train": [], "val": []}
    for context in contexts.values():
        for split in ("train", "val"):
            loader = context.loaders["%s_loader" % split]
            supports = [support.to(device) for support in context.supports]
            llm_encoding = context.llm_encoding.to(device)
            for batch in loader:
                x, label, batch_meta = split_batch(batch, getattr(loader, "batch_meta_keys", context.loaders.get("batch_meta_keys", [])))
                x = x.to(device)
                label = label.to(device)
                batch_meta = move_batch_meta(batch_meta, device)
                batch_meta.update({"output_len": int(output_len), "output_dim": int(output_dim)})
                baseline = backbone(x, supports, llm_encoding)
                batch_size = int(x.shape[0])
                loss_mask = torch.zeros_like(label, dtype=torch.bool)
                if context.known_set:
                    loss_mask[:, :, sorted(context.known_set), :] = True
                payloads[split].append(
                    {
                        "city": context.city,
                        "city_id": int(context.city_id),
                        "num_nodes": int(context.num_nodes),
                        "sample_ids": _sample_ids_from_meta(batch_meta, batch_size),
                        "history": x.detach().cpu(),
                        "label": label.detach().cpu(),
                        "baseline": baseline.detach().cpu(),
                        "label_denorm": _inverse_transform(label, context.scalers).detach().cpu(),
                        "baseline_denorm": _inverse_transform(baseline, context.scalers).detach().cpu(),
                        "loss_mask": loss_mask.detach().cpu(),
                        "null_value": float(context.null_value),
                    }
                )
    return payloads


def _valid_label_mask(label, null_value):
    null_value = float(null_value)
    if null_value != null_value:
        return ~torch.isnan(label)
    return label > null_value + 0.1


def _masked_abs_loss(prediction, label, loss_mask, null_value, available=None):
    mask = loss_mask.to(dtype=torch.bool, device=prediction.device)
    mask = mask & _valid_label_mask(label, null_value)
    if available is not None:
        availability = available.to(dtype=torch.bool, device=prediction.device).view(available.shape[0], 1, available.shape[1], 1)
        mask = mask & availability
    if not torch.any(mask):
        raise ValueError("no valid labels for multi-source expert loss")
    return (prediction - label).abs()[mask].mean()


def _macro_val_mae(losses):
    by_city = {}
    for city, value in losses:
        by_city.setdefault(city, []).append(float(value))
    per_city = {city: sum(values) / max(len(values), 1) for city, values in by_city.items()}
    macro = sum(per_city.values()) / max(len(per_city), 1)
    return macro, per_city


def _payload_keys_values(train_payloads, output_len, output_dim):
    keys = []
    values = []
    source_city_ids = []
    source_city_names = []
    sample_ids = []
    for item in train_payloads:
        history = item["history"].detach().cpu()
        label = item["label"].detach().cpu()
        batch_size, input_len, node_count, _ = history.shape
        keys.append(history[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(batch_size * node_count, input_len * output_dim))
        values.append(label[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(batch_size * node_count, int(output_len), int(output_dim)))
        for batch_idx in range(batch_size):
            source_city_ids.extend([int(item["city_id"])] * node_count)
            source_city_names.extend([item["city"]] * node_count)
            sample_ids.extend([int(item["sample_ids"][batch_idx])] * node_count)
    if not keys:
        raise ValueError("cannot build expert bank from empty payloads")
    return {
        "keys": torch.cat(keys, dim=0),
        "values": torch.cat(values, dim=0),
        "source_city_ids": torch.as_tensor(source_city_ids, dtype=torch.long),
        "source_city_names": list(source_city_names),
        "sample_ids": torch.as_tensor(sample_ids, dtype=torch.long),
    }


def build_itsc_bank(train_payloads, output_len, output_dim):
    merged = _payload_keys_values(train_payloads, output_len, output_dim)
    return {
        "global": {
            "dyn_keys": merged["keys"],
            "future": merged["values"],
            "sample_ids": merged["sample_ids"],
            "source_city_ids": merged["source_city_ids"],
            "source_city_names": merged["source_city_names"],
        },
        "metadata": {"output_len": int(output_len), "output_dim": int(output_dim), "bank_type": "multisource_itsc"},
    }


def build_raft_bank(train_payloads, output_len, output_dim):
    merged = _payload_keys_values(train_payloads, output_len, output_dim)
    source_to_id = {name: idx for idx, name in enumerate(sorted(set(merged["source_city_names"])))}
    source_ids = torch.as_tensor([source_to_id[name] for name in merged["source_city_names"]], dtype=torch.long)
    return {
        "keys": merged["keys"],
        "values": merged["values"],
        "sample_indices": merged["sample_ids"],
        "source_ids": source_ids,
        "source_to_id": source_to_id,
        "source_city_ids": merged["source_city_ids"],
        "source_city_names": merged["source_city_names"],
        "metadata": {"output_len": int(output_len), "output_dim": int(output_dim), "bank_type": "multisource_raft"},
    }


def build_tpb_bank(train_payloads, output_dim, patch_len, max_patterns=20000):
    patterns = []
    for item in train_payloads:
        history = item["history"].detach().cpu()[:, :, :, :output_dim]
        usable_len = (history.shape[1] // int(patch_len)) * int(patch_len)
        if usable_len <= 0:
            continue
        patches = history[:, :usable_len].reshape(
            history.shape[0],
            usable_len // int(patch_len),
            int(patch_len),
            history.shape[2],
            int(output_dim),
        )
        patches = patches.permute(0, 3, 1, 2, 4).reshape(-1, int(patch_len) * int(output_dim))
        patterns.append(patches)
    if not patterns:
        raise ValueError("cannot build TPB bank from empty payloads")
    patterns = torch.cat(patterns, dim=0)
    if int(max_patterns) > 0 and patterns.shape[0] > int(max_patterns):
        indices = torch.linspace(0, patterns.shape[0] - 1, steps=int(max_patterns)).long()
        patterns = patterns[indices]
    return {
        "patterns": patterns,
        "metadata": {
            "patch_len": int(patch_len),
            "output_dim": int(output_dim),
            "bank_type": "multisource_tpb",
            "num_patterns": int(patterns.shape[0]),
        },
    }


def _evaluate_prior_expert(expert, payloads, args, device, name):
    rows = []
    for split, items in (("train", payloads["train"]), ("val", payloads["val"])):
        losses = []
        for item in items:
            baseline = item["baseline"].to(device)
            output = expert.forward_correction(
                item["history"].to(device),
                supports=None,
                llm_encoding=None,
                batch_meta={"output_len": int(args.output_len), "output_dim": int(args.output_dim), "sample_ids": torch.as_tensor(item["sample_ids"], device=device)},
                baseline_pred=baseline,
            )
            loss = _masked_abs_loss(
                baseline + output.delta,
                item["label"].to(device),
                item["loss_mask"].to(device),
                item["null_value"],
                output.available.to(device),
            )
            losses.append((item["city"], float(loss.detach().cpu())))
        macro, by_city = _macro_val_mae(losses)
        row = {"expert": name, "epoch": 0, "%s_mae" % split: macro, "val_macro_mae": macro if split == "val" else ""}
        row.update({"%s_%s_mae" % (split, city): value for city, value in by_city.items()})
        rows.append(row)
    val_row = rows[-1]
    summary = {"expert": name, "epoch": 0, "val_macro_mae": float(val_row["val_macro_mae"])}
    summary.update({key: value for key, value in val_row.items() if key.startswith("val_") and key.endswith("_mae")})
    return summary, rows


def train_itsc_expert(args, payloads, run_dir, device):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bank = build_itsc_bank(payloads["train"], args.output_len, args.output_dim)
    bank_path = run_dir / "itsc_bank.pkl"
    with bank_path.open("wb") as handle:
        pickle.dump(bank, handle)
    config = _itsc_config(run_dir, args)
    expert = ITSCExpert().prepare(config, {"dataset": "multisource", "source_data": "multisource"})
    summary, rows = _evaluate_prior_expert(expert, payloads, args, device, "itsc")
    (run_dir / "itsc.json").write_text(json.dumps({"metrics": _jsonable(summary)}, indent=2, sort_keys=True), encoding="utf-8")
    _write_yaml(run_dir / "experts_itsc.yaml", {"experts": {"itsc": config}})
    return {"config": config, "summary": summary, "history": rows}


def _itsc_config(run_dir, args):
    return {
        "class": "ITSCExpert",
        "enabled": True,
        "mode": "prior",
        "bank_path": str(Path(run_dir) / "itsc_bank.pkl"),
        "top_k": int(args.itsc_top_k),
        "temperature": float(args.itsc_temperature),
        "scale": 1.0,
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


def train_raft_expert(args, payloads, run_dir, device):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bank = build_raft_bank(payloads["train"], args.output_len, args.output_dim)
    torch.save(bank, run_dir / "raft_bank.pt")
    base_config = _raft_config(run_dir, args, alpha=1.0)
    expert = RAFTExpert().prepare(base_config, {"dataset": "multisource", "source_data": "multisource"})
    train_cached = _prior_payload(expert, payloads["train"], args, device)
    val_cached = _prior_payload(expert, payloads["val"], args, device)
    alpha = TrainableAlpha(args.output_len, args.raft_alpha_mode).to(device)
    optimizer = torch.optim.Adam(alpha.parameters(), lr=float(args.raft_alpha_lr))
    best = {"val_macro_mae": float("inf"), "epoch": -1}
    rows = []
    stale = 0
    for epoch in range(int(args.max_epochs)):
        train_losses = []
        alpha.train()
        for item in train_cached:
            optimizer.zero_grad()
            baseline = item["baseline"].to(device)
            prediction = alpha(baseline, item["prior"].to(device), item["available"].to(device))
            pred_loss = _masked_abs_loss(prediction, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device))
            loss = pred_loss + float(args.residual_l1) * (prediction - baseline).abs().mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(pred_loss.detach().cpu()))
        val_losses = []
        alpha.eval()
        with torch.no_grad():
            for item in val_cached:
                pred = alpha(item["baseline"].to(device), item["prior"].to(device), item["available"].to(device))
                val_losses.append((item["city"], float(_masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device)).detach().cpu())))
        val_macro, val_by_city = _macro_val_mae(val_losses)
        row = {"expert": "raft", "epoch": epoch, "train_mae": sum(train_losses) / max(len(train_losses), 1), "val_macro_mae": val_macro}
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro + float(args.early_stop_min_delta) < best["val_macro_mae"]:
            best = dict(row)
            stale = 0
            (run_dir / "raft.json").write_text(json.dumps({"metrics": _jsonable(best), "alpha": _jsonable(alpha.export_alpha())}, indent=2, sort_keys=True), encoding="utf-8")
            _write_yaml(run_dir / "experts_raft.yaml", {"experts": {"raft": _raft_config(run_dir, args, alpha.export_alpha())}})
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    return {"config": _raft_config(run_dir, args, _alpha_from_json(run_dir / "raft.json")), "summary": best, "history": rows}


def _raft_config(run_dir, args, alpha):
    return {
        "class": "RAFTExpert",
        "enabled": True,
        "mode": "residual",
        "bank_path": str(Path(run_dir) / "raft_bank.pt"),
        "top_k": int(args.raft_top_k),
        "temperature": float(args.raft_temperature),
        "prior_alpha": _jsonable(alpha),
        "scale": 1.0,
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


def _alpha_from_json(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return json.loads(path.read_text(encoding="utf-8")).get("alpha", 0.0)


@torch.no_grad()
def _prior_payload(expert, payloads, args, device):
    cached = []
    for item in payloads:
        baseline = item["baseline"].to(device)
        output = expert.forward_correction(
            item["history"].to(device),
            supports=None,
            llm_encoding=None,
            batch_meta={"output_len": int(args.output_len), "output_dim": int(args.output_dim), "sample_ids": torch.as_tensor(item["sample_ids"], device=device)},
            baseline_pred=baseline,
        )
        cached.append({**item, "prior": (baseline + output.delta).detach().cpu(), "available": output.available.detach().cpu()})
    return cached


def train_tpb_expert(args, payloads, run_dir, device):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bank = build_tpb_bank(payloads["train"], args.output_dim, args.tpb_patch_len, args.tpb_max_patterns)
    torch.save(bank, run_dir / "tpb_bank.pt")
    config = _tpb_config(run_dir, args)
    expert = TPBExpert().prepare(config, {"dataset": "multisource", "source_data": "multisource"})
    summary, rows = _evaluate_prior_expert(expert, payloads, args, device, "tpb")
    (run_dir / "tpb.json").write_text(json.dumps({"metrics": _jsonable(summary)}, indent=2, sort_keys=True), encoding="utf-8")
    _write_yaml(run_dir / "experts_tpb.yaml", {"experts": {"tpb": config}})
    return {"config": config, "summary": summary, "history": rows}


def _tpb_config(run_dir, args):
    return {
        "class": "TPBExpert",
        "enabled": True,
        "mode": "prior",
        "bank_path": str(Path(run_dir) / "tpb_bank.pt"),
        "top_k": int(args.tpb_top_k),
        "temperature": float(args.tpb_temperature),
        "patch_len": int(args.tpb_patch_len),
        "scale": 1.0,
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


class TrainableScalarGate(torch.nn.Module):
    def __init__(self, initial_bias=-4.0):
        super().__init__()
        self.gate_bias = torch.nn.Parameter(torch.tensor(float(initial_bias), dtype=torch.float32))

    def forward(self, baseline, base_delta):
        return baseline + torch.sigmoid(self.gate_bias).to(dtype=baseline.dtype, device=baseline.device) * base_delta

    def export(self):
        return {"gate_scale": 1.0, "gate_bias": float(self.gate_bias.detach().cpu().item())}


def train_itsc_segment_gate_expert(args, payloads, run_dir, device, base_itsc_config):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = run_dir / "base_itsc_config.yaml"
    _write_yaml(base_config_path, {"experts": {"itsc": base_itsc_config}})
    base_expert = ITSCExpert().prepare(base_itsc_config, {"dataset": "multisource", "source_data": "multisource"})
    train_cached = _delta_payload(base_expert, payloads["train"], args, device)
    val_cached = _delta_payload(base_expert, payloads["val"], args, device)
    gate = TrainableScalarGate().to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=float(args.itsc_gate_lr))
    best = {"val_macro_mae": float("inf"), "epoch": -1}
    rows = []
    stale = 0
    for epoch in range(int(args.max_epochs)):
        train_losses = []
        gate.train()
        for item in train_cached:
            optimizer.zero_grad()
            pred = gate(item["baseline"].to(device), item["delta"].to(device))
            loss = _masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device))
            loss = loss + float(args.itsc_gate_l1) * torch.sigmoid(gate.gate_bias).abs()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_losses = []
        gate.eval()
        with torch.no_grad():
            for item in val_cached:
                pred = gate(item["baseline"].to(device), item["delta"].to(device))
                val_losses.append((item["city"], float(_masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device)).detach().cpu())))
        val_macro, val_by_city = _macro_val_mae(val_losses)
        row = {"expert": "itsc_segment_gate", "epoch": epoch, "train_mae": sum(train_losses) / max(len(train_losses), 1), "val_macro_mae": val_macro}
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro + float(args.early_stop_min_delta) < best["val_macro_mae"]:
            best = dict(row)
            stale = 0
            state = {key: torch.tensor([value], dtype=torch.float32) for key, value in gate.export().items()}
            torch.save(state, run_dir / "itsc_segment_gate.pt")
            (run_dir / "itsc_segment_gate.json").write_text(json.dumps({"metrics": _jsonable(best), **gate.export()}, indent=2, sort_keys=True), encoding="utf-8")
            _write_yaml(run_dir / "experts_itsc_segment_gate.yaml", {"experts": {"itsc_segment_gate": _itsc_segment_gate_config(run_dir, args)}})
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    return {"config": _itsc_segment_gate_config(run_dir, args), "summary": best, "history": rows}


@torch.no_grad()
def _delta_payload(expert, payloads, args, device):
    cached = []
    for item in payloads:
        baseline = item["baseline"].to(device)
        output = expert.forward_correction(
            item["history"].to(device),
            supports=None,
            llm_encoding=None,
            batch_meta={"output_len": int(args.output_len), "output_dim": int(args.output_dim), "sample_ids": torch.as_tensor(item["sample_ids"], device=device)},
            baseline_pred=baseline,
        )
        cached.append({**item, "delta": output.delta.detach().cpu(), "available": output.available.detach().cpu()})
    return cached


def _itsc_segment_gate_config(run_dir, args):
    summary_path = Path(run_dir) / "itsc_segment_gate.json"
    gate_bias = -4.0
    gate_scale = 1.0
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        gate_bias = float(payload.get("gate_bias", gate_bias))
        gate_scale = float(payload.get("gate_scale", gate_scale))
    return {
        "class": "ITSCSegmentGateExpert",
        "enabled": True,
        "mode": "segment_gate",
        "checkpoint_path": str(Path(run_dir) / "itsc_segment_gate.pt"),
        "base_itsc_config": str(Path(run_dir) / "base_itsc_config.yaml"),
        "hidden_dim": int(args.itsc_gate_hidden_dim),
        "gate_mode": "learned",
        "gate_scale": gate_scale,
        "gate_bias": gate_bias,
        "min_gate_available": float(args.min_gate_available),
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


class TrainableCalibration(torch.nn.Module):
    def __init__(self, output_len, output_dim, scope):
        super().__init__()
        size = 1 if scope == "shared" else int(output_len)
        self.scale = torch.nn.Parameter(torch.ones(size, int(output_dim)))
        self.bias = torch.nn.Parameter(torch.zeros(size, int(output_dim)))

    def forward(self, baseline):
        if self.scale.shape[0] == 1:
            scale = self.scale.view(1, 1, 1, -1)
            bias = self.bias.view(1, 1, 1, -1)
        else:
            scale = self.scale.view(1, self.scale.shape[0], 1, self.scale.shape[1])
            bias = self.bias.view(1, self.bias.shape[0], 1, self.bias.shape[1])
        return baseline * scale + bias


def train_calibration_expert(args, payloads, run_dir, device):
    model = TrainableCalibration(args.output_len, args.output_dim, args.calibration_scope).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.cal_lr))
    best = {"val_macro_mae": float("inf"), "epoch": -1}
    rows = []
    stale = 0
    for epoch in range(int(args.max_epochs)):
        model.train()
        train_losses = []
        for item in payloads["train"]:
            baseline = item["baseline"].to(device)
            label = item["label"].to(device)
            optimizer.zero_grad()
            prediction = model(baseline)
            pred_loss = _masked_abs_loss(prediction, label, item["loss_mask"].to(device), item["null_value"])
            identity = (model.scale - 1.0).abs().mean() + model.bias.abs().mean()
            residual = (prediction - baseline).abs().mean()
            loss = pred_loss + float(args.identity_l1) * identity + float(args.residual_l1) * residual
            loss.backward()
            optimizer.step()
            train_losses.append(float(pred_loss.detach().cpu()))
        val_losses = []
        model.eval()
        with torch.no_grad():
            for item in payloads["val"]:
                pred = model(item["baseline"].to(device))
                val_losses.append((item["city"], float(_masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"]).detach().cpu())))
        val_macro, val_by_city = _macro_val_mae(val_losses)
        row = {"expert": "calibration", "epoch": epoch, "train_mae": sum(train_losses) / max(len(train_losses), 1), "val_macro_mae": val_macro}
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro + float(args.early_stop_min_delta) < best["val_macro_mae"]:
            best = dict(row)
            stale = 0
            _write_calibration_artifacts(run_dir, model.state_dict(), args, best)
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    return {"config": _calibration_config(run_dir, args), "summary": best, "history": rows}


def _write_calibration_artifacts(run_dir, state_dict, args, metrics):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(state, run_dir / "calibration.pt")
    (run_dir / "calibration.json").write_text(json.dumps({"metrics": _jsonable(metrics)}, indent=2, sort_keys=True), encoding="utf-8")
    _write_yaml(run_dir / "experts_calibration.yaml", {"experts": {"calibration": _calibration_config(run_dir, args)}})


def _calibration_config(run_dir, args):
    return {
        "class": "CalibrationResidualExpert",
        "enabled": True,
        "mode": "learned_bias",
        "checkpoint_path": str(Path(run_dir) / "calibration.pt"),
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


def build_source_window_bank(train_payloads, output_len, output_dim):
    keys = []
    values = []
    source_city_ids = []
    source_city_names = []
    sample_ids = []
    for item in train_payloads:
        history = item["history"].detach().cpu()
        label = item["label"].detach().cpu()
        batch_size, input_len, node_count, _ = history.shape
        keys.append(history[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(batch_size * node_count, input_len * output_dim))
        values.append(label[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(batch_size * node_count, int(output_len), int(output_dim)))
        for batch_idx in range(batch_size):
            source_city_ids.extend([int(item["city_id"])] * node_count)
            source_city_names.extend([item["city"]] * node_count)
            sample_ids.extend([int(item["sample_ids"][batch_idx])] * node_count)
    if not keys:
        raise ValueError("cannot build source_window bank from empty payloads")
    return {
        "keys": torch.cat(keys, dim=0),
        "values": torch.cat(values, dim=0),
        "source_city_ids": torch.as_tensor(source_city_ids, dtype=torch.long),
        "source_city_names": list(source_city_names),
        "sample_ids": torch.as_tensor(sample_ids, dtype=torch.long),
        "metadata": {"output_len": int(output_len), "output_dim": int(output_dim)},
    }


class TrainableAlpha(torch.nn.Module):
    def __init__(self, output_len, mode):
        super().__init__()
        size = 1 if mode == "scalar" else int(output_len)
        self.logit_alpha = torch.nn.Parameter(torch.full((size,), -2.0))

    def forward(self, baseline, prior, available):
        alpha = torch.sigmoid(self.logit_alpha)
        shaped = alpha.view(1, 1, 1, 1) if alpha.numel() == 1 else alpha.view(1, alpha.numel(), 1, 1)
        return baseline + shaped * (prior - baseline) * available.view(available.shape[0], 1, available.shape[1], 1).to(baseline.dtype)

    def export_alpha(self):
        alpha = torch.sigmoid(self.logit_alpha.detach()).cpu()
        return float(alpha.item()) if alpha.numel() == 1 else alpha.tolist()


@torch.no_grad()
def _source_window_prior_payload(payloads, bank, args, device):
    expert = SourceWindowExpert()
    expert.prepare(
        {
            "mode": "source_window",
            "bank": bank,
            "top_k": int(args.source_top_k),
            "temperature": float(args.source_temperature),
            "confidence_threshold": float(args.source_confidence_threshold),
            "min_top1": float(args.min_top1),
            "min_margin": float(args.min_margin),
            "max_entropy": float(args.max_entropy),
            "score_chunk_size": int(args.score_chunk_size),
            "alpha": 1.0,
            "output_len": int(args.output_len),
            "output_dim": int(args.output_dim),
        },
        {"dataset": "multisource", "source_data": "multisource"},
    )
    cached = []
    for item in payloads:
        baseline = item["baseline"].to(device)
        output = expert.forward_correction(
            item["history"].to(device),
            batch_meta={"output_len": int(args.output_len), "output_dim": int(args.output_dim)},
            baseline_pred=baseline,
        )
        cached.append({**item, "prior": output.raw_prior.detach().cpu(), "available": output.available.detach().cpu()})
    return cached


def train_source_window_expert(args, payloads, run_dir, device):
    bank = build_source_window_bank(payloads["train"], args.output_len, args.output_dim)
    train_cached = _source_window_prior_payload(payloads["train"], bank, args, device)
    val_cached = _source_window_prior_payload(payloads["val"], bank, args, device)
    alpha = TrainableAlpha(args.output_len, args.source_alpha_mode).to(device)
    optimizer = torch.optim.Adam(alpha.parameters(), lr=float(args.source_alpha_lr))
    best = {"val_macro_mae": float("inf"), "epoch": -1}
    rows = []
    stale = 0
    for epoch in range(int(args.max_epochs)):
        train_losses = []
        alpha.train()
        for item in train_cached:
            optimizer.zero_grad()
            baseline = item["baseline"].to(device)
            prediction = alpha(baseline, item["prior"].to(device), item["available"].to(device))
            pred_loss = _masked_abs_loss(prediction, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device))
            residual = (prediction - baseline).abs().mean()
            loss = pred_loss + float(args.residual_l1) * residual
            loss.backward()
            optimizer.step()
            train_losses.append(float(pred_loss.detach().cpu()))
        val_losses = []
        alpha.eval()
        with torch.no_grad():
            for item in val_cached:
                pred = alpha(item["baseline"].to(device), item["prior"].to(device), item["available"].to(device))
                val_losses.append((item["city"], float(_masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], item["available"].to(device)).detach().cpu())))
        val_macro, val_by_city = _macro_val_mae(val_losses)
        row = {"expert": "source_window", "epoch": epoch, "train_mae": sum(train_losses) / max(len(train_losses), 1), "val_macro_mae": val_macro}
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro + float(args.early_stop_min_delta) < best["val_macro_mae"]:
            best = dict(row)
            stale = 0
            _write_source_window_artifacts(run_dir, alpha.state_dict(), alpha.export_alpha(), bank, args, best)
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    return {"config": _source_window_config(run_dir, args, _alpha_from_state_or_default(run_dir)), "summary": best, "history": rows}


def _write_source_window_artifacts(run_dir, state_dict, alpha, bank, args, metrics):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, run_dir / "source_window.pt")
    torch.save(bank, run_dir / "source_window_bank.pt")
    (run_dir / "source_window.json").write_text(json.dumps({"metrics": _jsonable(metrics), "alpha": _jsonable(alpha)}, indent=2, sort_keys=True), encoding="utf-8")
    _write_yaml(run_dir / "experts_source_window.yaml", {"experts": {"source_window": _source_window_config(run_dir, args, alpha)}})


def _alpha_from_state_or_default(run_dir):
    summary_path = Path(run_dir) / "source_window.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8")).get("alpha", 0.0)
    return 0.0


def _source_window_config(run_dir, args, alpha):
    return {
        "class": "SourceWindowExpert",
        "enabled": True,
        "mode": "source_window",
        "checkpoint_path": str(Path(run_dir) / "source_window.pt"),
        "bank_path": str(Path(run_dir) / "source_window_bank.pt"),
        "top_k": int(args.source_top_k),
        "temperature": float(args.source_temperature),
        "confidence_threshold": float(args.source_confidence_threshold),
        "min_top1": float(args.min_top1),
        "min_margin": float(args.min_margin),
        "max_entropy": float(args.max_entropy),
        "score_chunk_size": int(args.score_chunk_size),
        "alpha": _jsonable(alpha),
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


def _volatility_delta(module, history, baseline, args):
    features = history_node_features(history).to(dtype=next(module.parameters()).dtype, device=next(module.parameters()).device)
    horizon = horizon_fraction(baseline).to(dtype=features.dtype, device=features.device)
    raw_delta = module(features, horizon)
    return float(args.max_abs_delta) * torch.tanh(raw_delta).to(dtype=baseline.dtype, device=baseline.device).expand_as(baseline)


def _volatility_available(history, reference, args):
    features = history_node_features(history).to(device=reference.device)
    available = torch.ones(features.shape[0], features.shape[1], dtype=torch.bool, device=reference.device)
    if float(args.min_history_std) > 0.0:
        available = available & (features[:, :, 1] >= float(args.min_history_std))
    if float(args.min_history_max) > 0.0:
        available = available & (features[:, :, 2] >= float(args.min_history_max))
    return available


def train_volatility_peak_expert(args, payloads, run_dir, device):
    model = VolatilityPeakResidual(feature_dim=6, hidden_dim=int(args.volatility_hidden_dim), output_len=int(args.output_len)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.base_lr))
    best = {"val_macro_mae": float("inf"), "epoch": -1}
    rows = []
    stale = 0
    for epoch in range(int(args.max_epochs)):
        train_losses = []
        model.train()
        for item in payloads["train"]:
            optimizer.zero_grad()
            baseline = item["baseline"].to(device)
            delta = _volatility_delta(model, item["history"].to(device), baseline, args)
            available = _volatility_available(item["history"].to(device), baseline, args)
            prediction = baseline + delta
            pred_loss = _masked_abs_loss(prediction, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], available)
            identity = sum(torch.mean(parameter ** 2) for parameter in model.parameters())
            max_delta_penalty = torch.relu(torch.abs(delta) - float(args.max_abs_delta)).mean()
            loss = pred_loss + float(args.identity_l2) * identity + float(args.max_delta_penalty) * max_delta_penalty + float(args.residual_l1) * torch.abs(delta).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(pred_loss.detach().cpu()))
        val_losses = []
        model.eval()
        with torch.no_grad():
            for item in payloads["val"]:
                baseline = item["baseline"].to(device)
                available = _volatility_available(item["history"].to(device), baseline, args)
                pred = baseline + _volatility_delta(model, item["history"].to(device), baseline, args)
                val_losses.append((item["city"], float(_masked_abs_loss(pred, item["label"].to(device), item["loss_mask"].to(device), item["null_value"], available).detach().cpu())))
        val_macro, val_by_city = _macro_val_mae(val_losses)
        row = {"expert": "volatility_peak", "epoch": epoch, "train_mae": sum(train_losses) / max(len(train_losses), 1), "val_macro_mae": val_macro}
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro + float(args.early_stop_min_delta) < best["val_macro_mae"]:
            best = dict(row)
            stale = 0
            _write_volatility_artifacts(run_dir, model.state_dict(), args, best)
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    return {"config": _volatility_config(run_dir, args), "summary": best, "history": rows}


def _write_volatility_artifacts(run_dir, state_dict, args, metrics):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in state_dict.items()}, run_dir / "volatility_peak.pt")
    (run_dir / "volatility_peak.json").write_text(json.dumps({"metrics": _jsonable(metrics)}, indent=2, sort_keys=True), encoding="utf-8")
    _write_yaml(run_dir / "experts_volatility_peak.yaml", {"experts": {"volatility_peak": _volatility_config(run_dir, args)}})


def _volatility_config(run_dir, args):
    return {
        "class": "VolatilityPeakExpert",
        "enabled": True,
        "mode": "learned_gate",
        "checkpoint_path": str(Path(run_dir) / "volatility_peak.pt"),
        "hidden_dim": int(args.volatility_hidden_dim),
        "max_abs_delta": float(args.max_abs_delta),
        "min_history_std": float(args.min_history_std),
        "min_history_max": float(args.min_history_max),
        "output_len": int(args.output_len),
        "output_dim": int(args.output_dim),
    }


def _enabled_experts(value):
    experts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not experts or experts == ["all"]:
        experts = list(ALL_MULTISOURCE_EXPERTS)
    unknown = [name for name in experts if name not in SUPPORTED_EXPERTS]
    if unknown:
        raise ValueError("unsupported multi-source experts %r; supported=%r" % (unknown, SUPPORTED_EXPERTS))
    return experts


def train_experts_multisource_from_contexts(args, backbone, contexts, device, source_cities, target_city):
    torch.manual_seed(int(args.seed))
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payloads = collect_expert_payloads(backbone, contexts, device, output_len=args.output_len, output_dim=args.output_dim)
    configs = {}
    summaries = {}
    history_rows = []
    base_itsc_config = None
    for expert_name in _enabled_experts(args.enabled_experts):
        expert_dir = run_path / expert_name
        if expert_name == "itsc":
            result = train_itsc_expert(args, payloads, expert_dir, device)
            base_itsc_config = result["config"]
        elif expert_name == "raft":
            result = train_raft_expert(args, payloads, expert_dir, device)
        elif expert_name == "tpb":
            result = train_tpb_expert(args, payloads, expert_dir, device)
        elif expert_name == "calibration":
            result = train_calibration_expert(args, payloads, expert_dir, device)
        elif expert_name == "source_window":
            result = train_source_window_expert(args, payloads, expert_dir, device)
        elif expert_name == "volatility_peak":
            result = train_volatility_peak_expert(args, payloads, expert_dir, device)
        elif expert_name == "itsc_segment_gate":
            if base_itsc_config is None:
                base_itsc_dir = run_path / "itsc"
                base_itsc_config = _itsc_config(base_itsc_dir, args)
                if not Path(base_itsc_config["bank_path"]).exists():
                    train_itsc_expert(args, payloads, base_itsc_dir, device)
            result = train_itsc_segment_gate_expert(args, payloads, expert_dir, device, base_itsc_config)
        else:
            raise AssertionError("unreachable expert %s" % expert_name)
        configs[expert_name] = result["config"]
        summaries[expert_name] = result["summary"]
        history_rows.extend(result["history"])

    _write_yaml(run_path / "experts_multisource.yaml", {"experts": configs})
    if history_rows:
        fieldnames = sorted({key for row in history_rows for key in row})
        with (run_path / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history_rows)
    summary = {
        "source_cities": list(source_cities),
        "target_city": target_city,
        "enabled_experts": _enabled_experts(args.enabled_experts),
        "experts": summaries,
        "artifacts": {
            "expert_config": str(run_path / "experts_multisource.yaml"),
            "history": str(run_path / "history.csv"),
        },
    }
    (run_path / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Train multi-source expert artifacts.")
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--target_city", required=True)
    parser.add_argument("--backbone_ckpt", required=True)
    parser.add_argument("--run_dir", default="results/training/experts")
    parser.add_argument("--enabled_experts", default="all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--residual_l1", type=float, default=0.0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--device", default="")
    return parser


def main():
    args = build_parser().parse_args()
    source_cities, target = split_source_target(args.cities, args.target_city)
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
    backbone = _load_backbone(args, device)
    summary = train_experts_multisource_from_contexts(args, backbone, contexts, device, protocol.source_cities or source_cities, target)
    print(json.dumps(_jsonable(summary), sort_keys=True))


if __name__ == "__main__":
    main()
