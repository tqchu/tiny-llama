"""Broadcast‑safe Rotary Positional Embedding (RoPE)."""
import torch

__all__ = ["rope_cache", "apply_rope"]

def rope_cache(seq_len: int, dim: int, theta: float, device):
    """
    Returns sin, cos tensors with shape (1, 1, T, dim//2) so they broadcast over
    batch (B) and head (H) dimensions of q/k which are (B, H, T, D).
    """
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device) / half))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)        # (T, half)
    sin = freqs.sin()[None, None, :, :]     # (1, 1, T, half)
    cos = freqs.cos()[None, None, :, :]
    return sin, cos

def apply_rope(x, sin, cos):
    """Apply RoPE to (B,H,T,D) tensor."""
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    x_rot = torch.cat((x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos), dim=-1)
    return x_rot