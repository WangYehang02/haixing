import torch
import torch.nn as nn

from fmgad.losses import conditional_flow_matching_loss, flow_matching_loss


class MLPFlowMatching(nn.Module):
    """MLP velocity field for flow matching."""

    def __init__(self, d_in, dim_t=512, cond_dim=None):
        super().__init__()
        self.dim_t = dim_t
        self.cond_dim = cond_dim

        self.map_time = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        if cond_dim is not None:
            self.map_proto = nn.Linear(cond_dim, dim_t, bias=False)
            self.proto_proj = nn.Linear(cond_dim, dim_t)

        self.proj_x = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x, t, context=None, proto_alpha=None):
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=x.dtype).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.dim() > 1:
            t = t.squeeze(-1) if t.shape[-1] == 1 else t.flatten()

        t = t.clamp(0.0, 1.0)
        t_emb = self.map_time(t * 1000.0)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape(*t_emb.shape)
        t_emb = self.time_embed(t_emb)
        x_proj = self.proj_x(x)

        if context is not None and self.cond_dim is not None:
            if proto_alpha is None:
                proto_alpha = 1.0
            proto_emb = self.proto_proj(context)
            h = x_proj + t_emb + proto_alpha * proto_emb
        else:
            h = x_proj + t_emb

        return self.mlp(h)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class FlowMatchingModel(nn.Module):
    """Thin wrapper that stores the velocity network."""

    def __init__(self, velocity_fn, hid_dim, sigma_min=0.01, sigma_max=1.0):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.velocity_fn = velocity_fn

    def forward(self, x_1, proto=None, proto_alpha=None):
        if proto is not None:
            loss = conditional_flow_matching_loss(
                self.velocity_fn, x_1, proto, t_sampling="logit_normal", reduction="mean"
            )
        else:
            graph_context = torch.zeros(1, x_1.shape[1], device=x_1.device)
            loss = flow_matching_loss(self.velocity_fn, x_1, graph_context, reduction="mean")

        with torch.no_grad():
            x_0 = torch.randn_like(x_1)
            reconstructed = sample_flow_matching(
                self.velocity_fn, x_0, num_steps=10, proto=proto, proto_alpha=proto_alpha
            )
        score = torch.sqrt(torch.sum((x_1 - reconstructed) ** 2, dim=1))
        return loss, score, reconstructed


def sample_flow_matching(velocity_net, x_0, num_steps=50, proto=None, proto_alpha=None):
    """Euler ODE sampler from noise toward data."""
    device = x_0.device
    batch_size = x_0.shape[0]
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    velocity_net.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)
            v_t = velocity_net(x_t, t, context=proto, proto_alpha=proto_alpha)
            x_t = x_t + dt * v_t
    return x_t


def sample_flow_matching_free(proto_net, free_net, x_0, num_steps=50, proto=None, proto_alpha=None, weight=None):
    """Euler sampling with blended prototype-conditioned and free velocities."""
    device = x_0.device
    batch_size = x_0.shape[0]
    dt = 1.0 / num_steps
    x_t = x_0.clone()

    proto_net.eval()
    free_net.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=x_0.dtype)
            v_proto = proto_net(x_t, t, context=proto, proto_alpha=proto_alpha)
            v_free = free_net(x_t, t, context=None, proto_alpha=None)
            v_combined = (1.0 + weight) * v_free - weight * v_proto
            x_t = x_t + dt * v_combined
    return x_t
