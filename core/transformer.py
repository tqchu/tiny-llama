import torch.nn as nn
from .rmsnorm import RMSNorm
from .attention import MHA
from .ffn import SwiGLU
from .config import Config

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = MHA(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x