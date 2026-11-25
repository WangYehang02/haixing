"""Bound flow MLP workspace while preserving the full-graph objective and RNG."""
import copy

import torch
from torch.utils.checkpoint import checkpoint

from fmgad.models.flow_matching import (
    MLPFlowMatching, sample_flow_matching as _dense_sample,
    sample_flow_matching_free as _dense_sample_free,
)


_DENSE_FORWARD = MLPFlowMatching.forward


def slice_rows(value, left, right, n):
    if torch.is_tensor(value) and value.ndim and value.shape[0] == n:
        return value[left:right]
    return value


def install(block_size=65536):
    def block_forward(model, x, t, context=None, proto_alpha=None):
        n = x.shape[0]
        if n <= block_size:
            return _DENSE_FORWARD(model, x, t, context, proto_alpha)

        def evaluate(nodes, times, prototypes):
            return _DENSE_FORWARD(model, nodes, times, prototypes, proto_alpha)

        scores = []
        for left in range(0, n, block_size):
            right = min(left + block_size, n)
            args = (x[left:right], slice_rows(t, left, right, n),
                    slice_rows(context, left, right, n))
            if torch.is_grad_enabled():
                scores.append(checkpoint(evaluate, *args, use_reentrant=False, preserve_rng_state=False))
            else:
                scores.append(evaluate(*args))
        return torch.cat(scores, dim=0)

    MLPFlowMatching.forward = block_forward

    from fmgad import detector
    detector.sample_flow_matching = sample_constant_time
    detector.sample_flow_matching_free = sample_free_constant_time


@torch.no_grad()
def sample_constant_time(velocity_net, x_0, num_steps=50, proto=None, proto_alpha=None):
    h = x_0.clone()
    velocity_net.eval()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = h.new_full((1,), i * dt)
        velocity = velocity_net(h, t, context=proto, proto_alpha=proto_alpha)
        h = h + dt * velocity
    return h


@torch.no_grad()
def sample_free_constant_time(proto_net, free_net, x_0, num_steps=50, proto=None,
                              proto_alpha=None, weight=None):
    h = x_0.clone()
    proto_net.eval()
    free_net.eval()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = h.new_full((1,), i * dt)
        v_proto = proto_net(h, t, context=proto, proto_alpha=proto_alpha)
        v_free = free_net(h, t)
        h = h + dt * ((1 + weight) * v_free - weight * v_proto)
    return h


def selftest():
    torch.manual_seed(630)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original = MLPFlowMatching(14, dim_t=64, cond_dim=7).to(device)
    model = copy.deepcopy(original)
    original_input = torch.randn(251, 14, device=device, requires_grad=True)
    inputs = original_input.detach().clone().requires_grad_()
    install(block_size=61)
    for time_value in (0.2, torch.rand(251, device=device), torch.rand(251, 1, device=device)):
        for context in (None, torch.rand(1, 7, device=device), torch.rand(251, 7, device=device)):
            old = _DENSE_FORWARD(original, original_input, time_value, context, 0.6)
            new = model(inputs, time_value, context, 0.6)
            torch.testing.assert_close(new, old, atol=3e-7, rtol=3e-6)
            grad_output = torch.randn_like(new)
            old_grad = torch.autograd.grad(old, (original_input, *original.parameters()), grad_output, allow_unused=True)
            new_grad = torch.autograd.grad(new, (inputs, *model.parameters()), grad_output, allow_unused=True)
            for expected, actual in zip(old_grad, new_grad):
                if expected is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
            with torch.no_grad():
                test = model(inputs, time_value, context, 0.6)
                torch.testing.assert_close(test, old, atol=3e-7, rtol=3e-6)
    noise = original_input.detach()
    proto = torch.randn(1, 7, device=device)
    for steps in (1, 5, 10):
        old = _dense_sample(original, noise, num_steps=steps, proto=proto, proto_alpha=.6)
        new = sample_constant_time(model, noise, num_steps=steps, proto=proto, proto_alpha=.6)
        torch.testing.assert_close(old, new, atol=5e-7, rtol=4e-6)
        old = _dense_sample_free(original, original, noise, num_steps=steps,
                                 proto=proto, proto_alpha=.6, weight=1.5)
        new = sample_free_constant_time(model, model, noise, num_steps=steps,
                                       proto=proto, proto_alpha=.6, weight=1.5)
        torch.testing.assert_close(old, new, atol=5e-7, rtol=4e-6)
    print(f'OK split MLP, gradients, and broadcast-time Euler sampler on {device}')


if __name__ == '__main__':
    selftest()
