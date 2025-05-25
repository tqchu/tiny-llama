import torch
import torch.nn as nn
from .config import Config
from .transformer import Block
from .rmsnorm import RMSNorm

class TinyLlama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
    @classmethod
    def from_config(cls, path):
        return cls(Config.from_json(path))
    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))