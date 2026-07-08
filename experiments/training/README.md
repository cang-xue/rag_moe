# Multi-Source Zero-Shot RAG-MoE Training

This package keeps the complete training logic for multi-source zero-shot
transfer into a held-out delivery city.

## Protocol

Default cities:

```text
Delivery_SH, Delivery_HZ, Delivery_CQ, Delivery_YT, Delivery_JL
```

Pick one `target_city`. All other cities become source cities. The target city
must not be used for training, validation, DPO pair construction, reward
calculation, checkpoint selection, hyperparameter selection, or expert choice.
It is used only in the final zero-shot evaluation stage.

## Stages

1. Train a multi-source IMPEL backbone on source cities.
2. Train or build source-only expert artifacts.
3. Cache source-side candidates:
   - `none = backbone_pred`
   - `expert = backbone_pred + expert_delta`
4. Train a supervised router with frozen backbone and experts.
5. DPO-tune router heads with frozen backbone, experts, and router encoder.
6. Evaluate on the held-out target city using soft and guarded-hard routing.

## Dry-Run Plan

```powershell
python experiments/training/run_multisource_dpo_pipeline.py `
  --target_city Delivery_HZ `
  --run_tag hz_leave_one_out `
  --dry_run
```

This writes:

```text
results/training/<run_tag>/pipeline_plan.json
results/training/<run_tag>/pipeline_plan.txt
```

## DPO Router Tuning

The currently runnable training stage is DPO router-head tuning from a candidate
cache. The cache must contain:

```text
features: [B, N, F]
available: [B, N, E]
candidate_errors: [B, N, E]
candidate_names: list[str]
```

Run:

```powershell
python experiments/training/train_router_dpo.py `
  --candidate_cache results/training/<run_tag>/candidate_cache/source_candidates.pt `
  --router_ckpt results/training/<run_tag>/supervised_router/best_router.pt `
  --run_dir results/training/<run_tag>/dpo_router `
  --train_scope heads
```

DPO pair construction uses `sample x node` preference pairs. If an expert
clearly beats `none`, the pair is `expert > none`; otherwise the pair is
`none > best_non_none`. This keeps the backbone fallback explicit.

## Implementation Notes

The existing IMPEL code is mostly single-city. A true multi-source backbone and
multi-source expert builder still need city-aware loaders and artifact builders
that preserve each city's scaler, graph, LLM embeddings, and node count. This
package provides the protocol, dry-run plan, candidate/DPO utilities, and
runnable DPO tuning stage without weakening the zero-shot constraint.

