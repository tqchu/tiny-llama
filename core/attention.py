import math, torch, torch.nn as nn, torch.nn.functional as F
from .rotary import rope_cache, apply_rope
from .config import Config

class MHA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, HK = cfg.n_heads, cfg.n_kv_heads
        D = cfg.d_model // H
        self.d_head = D
        self.q_proj = nn.Linear(cfg.d_model, H * D, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, HK * D, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, HK * D, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.d_model, bias=False)
        self.sin = self.cos = None

    def _build_rope(self, T, device):
        if self.sin is None or self.sin.size(1) < T:
            self.sin, self.cos = rope_cache(T, self.d_head, self.cfg.rope_theta, device)
        return self.sin[:, :T], self.cos[:, :T]

    def forward(self, x):
        B, T, _ = x.size()
        H, HK, D = self.cfg.n_heads, self.cfg.n_kv_heads, self.d_head
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, HK, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, HK, D).transpose(1, 2)
        sin, cos = self._build_rope(T, x.device)
        q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)
        # repeat kv heads
        repeat = H // HK
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.full((T, T), float('-inf'), device=x.device)
        mask = torch.triu(mask, 1)  # zeros on & below diag
        attn_scores = attn_scores + mask
        attn = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)