import math
import torch
import torch.nn.functional as F


def _velocity_mse(prediction, target, residual_loss_weight):
    errors = F.mse_loss(prediction, target, reduction="none")
    if residual_loss_weight == 1.0:
        return errors.mean(dim=1)
    if not math.isfinite(residual_loss_weight) or residual_loss_weight < 0:
        raise ValueError("residual_loss_weight must be finite and nonnegative")
    if errors.size(1) % 2:
        raise ValueError("Expected equal-width embedding and residual blocks")
    embedding, residual = errors.chunk(2, dim=1)
    return (embedding.mean(dim=1) + residual_loss_weight * residual.mean(dim=1)) / (1.0 + residual_loss_weight)


def flow_matching_loss(model, x_1, graph_context, reduction="mean", weight=None, *, residual_loss_weight=1.0):
    """Flow-matching MSE on a linear path with uniform time sampling."""
    batch_size = x_1.shape[0]
    t = torch.rand(batch_size, 1, device=x_1.device)
    x_0 = torch.randn_like(x_1)
    x_t = (1 - t) * x_0 + t * x_1
    target_v = x_1 - x_0

    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)
        elif graph_context.shape[0] != batch_size:
            raise ValueError(f"Context shape mismatch: {graph_context.shape[0]} vs batch_size {batch_size}")

    pred_v = model(x_t, t.squeeze(-1) if t.dim() > 1 else t, graph_context)
    mse_per_sample = _velocity_mse(pred_v, target_v, residual_loss_weight)

    if weight is not None:
        mse_per_sample = mse_per_sample * weight

    if reduction == "mean":
        return mse_per_sample.mean()
    if reduction == "none":
        return mse_per_sample
    raise ValueError(f"Unknown reduction: {reduction}")


def conditional_flow_matching_loss(
    model, x_1, graph_context, t_sampling="uniform", reduction="mean", weight=None, *, proto_alpha=None,
    residual_loss_weight=1.0,
):
    """Conditional flow-matching loss with uniform or logit-normal time sampling."""
    batch_size = x_1.shape[0]

    if t_sampling == "uniform":
        t = torch.rand(batch_size, 1, device=x_1.device)
    elif t_sampling == "logit_normal":
        t_normal = torch.randn(batch_size, 1, device=x_1.device)
        t = torch.sigmoid(t_normal * 2.0)
    else:
        raise ValueError(f"Unknown t_sampling: {t_sampling}")

    x_0 = torch.randn_like(x_1)
    x_t = (1 - t) * x_0 + t * x_1
    target_v = x_1 - x_0

    if graph_context.dim() == 2:
        if graph_context.shape[0] == 1 and batch_size > 1:
            graph_context = graph_context.repeat(batch_size, 1)

    t_input = t.squeeze(-1) if t.dim() > 1 else t
    if proto_alpha is None:
        pred_v = model(x_t, t_input, graph_context)
    else:
        pred_v = model(x_t, t_input, graph_context, proto_alpha=proto_alpha)
    mse_per_sample = _velocity_mse(pred_v, target_v, residual_loss_weight)

    if weight is not None:
        mse_per_sample = mse_per_sample * weight

    if reduction == "mean":
        return mse_per_sample.mean()
    if reduction == "none":
        return mse_per_sample
    raise ValueError(f"Unknown reduction: {reduction}")
