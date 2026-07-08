import torch


def fuse_candidates(candidates, weights):
    if not isinstance(candidates, torch.Tensor):
        raise ValueError("candidates must be a torch.Tensor")
    if not isinstance(weights, torch.Tensor):
        raise ValueError("weights must be a torch.Tensor")
    if candidates.dim() != 5:
        raise ValueError(
            "candidates shape %r must be [B, T, E, N, C]"
            % (tuple(candidates.shape),)
        )
    if weights.dim() != 3:
        raise ValueError(
            "weights shape %r must be [B, N, E]" % (tuple(weights.shape),)
        )

    batch_size, _, expert_count, node_count, _ = candidates.shape
    weight_batch, weight_nodes, weight_experts = weights.shape
    if weight_batch != batch_size:
        raise ValueError(
            "weights batch dimension %r does not match candidates batch %r"
            % (weight_batch, batch_size)
        )
    if weight_experts != expert_count:
        raise ValueError(
            "weights expert dimension %r does not match candidates experts %r"
            % (weight_experts, expert_count)
        )
    if weight_nodes != node_count:
        raise ValueError(
            "weights node dimension %r does not match candidates nodes %r"
            % (weight_nodes, node_count)
        )

    broadcast_weights = weights.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)
    return (candidates * broadcast_weights).sum(dim=2)


def fuse_residuals(deltas, weights):
    if not isinstance(deltas, torch.Tensor):
        raise ValueError("deltas must be a torch.Tensor")
    if not isinstance(weights, torch.Tensor):
        raise ValueError("weights must be a torch.Tensor")
    if deltas.dim() != 5:
        raise ValueError(
            "deltas shape %r must be [B, T, E, N, C]"
            % (tuple(deltas.shape),)
        )
    if weights.dim() != 3:
        raise ValueError(
            "weights shape %r must be [B, N, E]" % (tuple(weights.shape),)
        )

    batch_size, _, expert_count, node_count, _ = deltas.shape
    weight_batch, weight_nodes, weight_experts = weights.shape
    if weight_batch != batch_size:
        raise ValueError(
            "weights batch dimension %r does not match deltas batch %r"
            % (weight_batch, batch_size)
        )
    if weight_experts != expert_count:
        raise ValueError(
            "weights expert dimension %r does not match deltas experts %r"
            % (weight_experts, expert_count)
        )
    if weight_nodes != node_count:
        raise ValueError(
            "weights node dimension %r does not match deltas nodes %r"
            % (weight_nodes, node_count)
        )

    broadcast_weights = weights.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)
    return (deltas * broadcast_weights).sum(dim=2)
