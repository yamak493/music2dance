"""Last-resort pure-PyTorch replacement for ``mamba_ssm.Mamba`` (no CUDA extension needed).

The module keeps exactly the parameter names/shapes of ``mamba_ssm.modules.mamba_simple.Mamba``
(in_proj, conv1d, x_proj, dt_proj, A_log, D, out_proj), so FlowerDance checkpoints load with
``strict=True``. The forward pass implements the same math as mamba_ssm's slow path
(``selective_scan_ref``) but evaluates the recurrence chunk-wise:

    inside a chunk:   h_t = exp(S_t) h_0 + sum_{s<=t} exp(S_t - S_s) dt_s B_s u_s,   S_t = sum_{k<=t} dt_k A
    across chunks:    the final state of a chunk is the initial state of the next one.

All exponents are <= 0, so it is numerically stable. It is slower than the fused CUDA kernel but
fast enough for FlowerDance's few sampling steps.

``install()`` registers a fake ``mamba_ssm`` package exposing this class so that
``from mamba_ssm import Mamba`` (used by FlowerDance/model/model.py) resolves to it.
"""
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK = 32          # time steps per chunk (memory ~ B*D*N*CHUNK^2*GROUP_CHUNKS floats)
GROUP_CHUNKS = 8    # chunks processed in parallel for the intra-chunk part


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                           chunk=CHUNK, group_chunks=GROUP_CHUNKS):
    """Same signature/semantics as mamba_ssm.ops.selective_scan_interface.selective_scan_ref
    for real-valued A (d, n) and input-dependent B, C of shape (b, n, l)."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    A = A.float()
    B = B.float()
    C = C.float()
    b, d, L = u.shape
    n = A.shape[1]
    pad = (-L) % chunk
    if pad:
        # Zero delta => no state change and zero input on padded steps (they are cut afterwards).
        u = F.pad(u, (0, pad))
        delta = F.pad(delta, (0, pad))
        B = F.pad(B, (0, pad))
        C = F.pad(C, (0, pad))
    Lp = L + pad
    nc = Lp // chunk
    # (b, d, nc, T) / (b, n, nc, T)
    u_c = u.view(b, d, nc, chunk)
    dt_c = delta.view(b, d, nc, chunk)
    B_c = B.view(b, n, nc, chunk)
    C_c = C.view(b, n, nc, chunk)

    causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=u.device))
    y = torch.empty(b, d, nc, chunk, dtype=torch.float32, device=u.device)
    h = torch.zeros(b, d, n, dtype=torch.float32, device=u.device)
    for g0 in range(0, nc, group_chunks):
        g1 = min(nc, g0 + group_chunks)
        dt = dt_c[:, :, g0:g1]                                    # (b,d,G,T)
        S = torch.cumsum(dt.unsqueeze(-1) * A[None, :, None, None, :], dim=3)  # (b,d,G,T,n)
        dBu = (dt * u_c[:, :, g0:g1]).unsqueeze(-1) * B_c[:, :, g0:g1].permute(0, 2, 3, 1).unsqueeze(1)  # (b,d,G,T,n)
        # W[t, s] = exp(S_t - S_s) for s <= t
        diff = S.unsqueeze(4) - S.unsqueeze(3)                    # (b,d,G,T(t),T(s),n)
        diff = diff.masked_fill(~causal[None, None, None, :, :, None], float("-inf"))
        h_local = torch.einsum("bdgtsn,bdgsn->bdgtn", torch.exp(diff), dBu)  # zero initial state
        del diff
        decay_in = torch.exp(S)                                   # (b,d,G,T,n) decay from chunk start
        Cg = C_c[:, :, g0:g1].permute(0, 2, 3, 1)                # (b,G,T,n)
        # Sequential carry of the state across the chunks of this group (cheap: b*d*n per chunk).
        for gi in range(g1 - g0):
            hs = h_local[:, :, gi] + decay_in[:, :, gi] * h.unsqueeze(2)   # (b,d,T,n)
            y[:, :, g0 + gi] = torch.einsum("bdtn,btn->bdt", hs, Cg[:, gi])
            h = hs[:, :, -1]
        del h_local, decay_in
    y = y.reshape(b, d, Lp)[..., :L]
    u = u[..., :L]
    out = y if D is None else y + u * D.float()[:, None]
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype_in)


def selective_scan_sequential(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """Straight copy of mamba_ssm's selective_scan_ref math (real A, variable B/C). Used for testing."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D[:, None]
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype_in)


class Mamba(nn.Module):
    """Parameter-compatible pure-PyTorch version of mamba_ssm.modules.mamba_simple.Mamba (no caching)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False,
                 use_fast_path=True, layer_idx=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=conv_bias, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, **factory_kwargs)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.scan_fn = selective_scan_chunked

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape
        xz = (hidden_states @ self.in_proj.weight.T).transpose(1, 2)  # (b, 2*d_inner, l)
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.to(xz.dtype)[:, None]
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])
        x_dbl = self.x_proj(x.transpose(1, 2).reshape(batch * seqlen, -1))
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (self.dt_proj.weight @ dt.T).reshape(self.d_inner, batch, seqlen).permute(1, 0, 2)
        Bm = Bm.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        Cm = Cm.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        y = self.scan_fn(x, dt, A, Bm, Cm, self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(),
                         delta_softplus=True)
        return self.out_proj(y.transpose(1, 2))


def install():
    """Register a fake `mamba_ssm` package whose `Mamba` is the pure-PyTorch class above."""
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = []
    pkg.__version__ = "pure-pytorch-fallback"
    pkg.Mamba = Mamba
    modules = types.ModuleType("mamba_ssm.modules")
    modules.__path__ = []
    simple = types.ModuleType("mamba_ssm.modules.mamba_simple")
    simple.Mamba = Mamba
    sys.modules["mamba_ssm"] = pkg
    sys.modules["mamba_ssm.modules"] = modules
    sys.modules["mamba_ssm.modules.mamba_simple"] = simple
    return Mamba
