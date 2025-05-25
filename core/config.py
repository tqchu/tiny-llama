"""TinyLlama configuration dataclass.

The public `config.json` on HuggingFace follows the Llama‑2 naming scheme.
This helper maps those keys → the concise internal names we use in code.
"""
from dataclasses import dataclass
import json

@dataclass
class Config:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    d_ff: int
    rope_theta: float
    rms_eps: float

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str):
        """Load HF‑style JSON and massage the field names."""
        with open(path) as f:
            j = json.load(f)
        # Map LLama → internal ------------------------------------------------
        return cls(
            vocab_size=j["vocab_size"],
            max_seq_len=j["max_position_embeddings"],
            d_model=j["hidden_size"],
            n_heads=j["num_attention_heads"],
            n_kv_heads=j.get("num_key_value_heads", j["num_attention_heads"] // 2),
            n_layers=j["num_hidden_layers"],
            d_ff=j["intermediate_size"],
            rope_theta=j.get("rope_theta", 10000.0),
            rms_eps=j.get("rms_norm_eps", 1e-5),
        )