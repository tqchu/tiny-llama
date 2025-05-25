"""Feed‑forward block that exactly matches Llama / TinyLlama layout.
   y = W_out( silu(W_gate x) * (W_up x) )
"""
import torch.nn as nn, torch.nn.functional as F
from .config import Config

class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # gate_proj
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # up_proj
        self.down = nn.Linear(cfg.d_ff,  cfg.d_model, bias=False) # down_proj
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))