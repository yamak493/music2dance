import torch

from m2d.mamba_fallback import Mamba, selective_scan_chunked, selective_scan_sequential


def test_chunked_scan_matches_reference():
    torch.manual_seed(0)
    b, d, n, L = 2, 24, 16, 203
    args = (torch.randn(b, d, L), torch.randn(b, d, L), -torch.exp(torch.randn(d, n)), torch.randn(b, n, L),
            torch.randn(b, n, L), torch.randn(d), torch.randn(b, d, L), torch.randn(d))
    ref = selective_scan_sequential(*args, delta_softplus=True)
    for chunk, group in [(32, 8), (16, 3), (64, 1), (7, 5)]:
        out = selective_scan_chunked(*args, delta_softplus=True, chunk=chunk, group_chunks=group)
        assert (out - ref).abs().max() / ref.abs().max() < 1e-5


def test_parameter_names_match_mamba_ssm():
    names = set(dict(Mamba(64).named_parameters()))
    assert names == {"A_log", "D", "in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight",
                     "dt_proj.weight", "dt_proj.bias", "out_proj.weight"}
