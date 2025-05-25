import torch, math
from transformers import AutoModelForCausalLM
from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.hf_state import load_state

CKPT = "/Users/chutruong/Academic/GraduationThesis/Projects/ipen-agent/llm/tinyllama/pretrained_1.1b"  # adjust to your path

hf = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).eval()
me = TinyLlama.from_config(f"{CKPT}/config.json").eval()
me.load_state_dict(load_state(CKPT), strict=True)

B, T, D = 2, 16, 2048
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)

for i in range(len(hf.model.layers)):
    print("-" * 80)
    print(f"Layer {i}")

    x = torch.randn(B, T, D)

    # STEP 0 — RMSNorm
    hf_norm = hf.model.layers[i].input_layernorm(x)
    my_norm = me.blocks[i].attn_norm(x)
    print(" STEP 0 RMSNorm max |Δ| =", (hf_norm - my_norm).abs().max().item())

    # STEP 1 — Q/K/V/O weights
    for name in ("q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight"):
        w_my = dict(me.blocks[i].attn.named_parameters())[name]
        w_hf = dict(hf.model.layers[i].self_attn.named_parameters())[name]
        print(f" STEP 1 {name} max |Δ| =", (w_my - w_hf).abs().max().item())

    # STEP 2 — RoPE sin/cos
    rotary   = hf.model.layers[i].self_attn.rotary_emb
    inv_freq = rotary.inv_freq
    t        = torch.arange(T, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs    = torch.outer(t, inv_freq)
    sin_hf   = freqs.sin()[None, None, :, :]
    cos_hf   = freqs.cos()[None, None, :, :]
    sin_my, cos_my = me.blocks[i].attn.build_rope(T, x.device)
    print(" STEP 2 sin max |Δ| =", (sin_hf - sin_my).abs().max().item())
    print(" STEP 2 cos max |Δ| =", (cos_hf - cos_my).abs().max().item())

    # STEP 3 — full block forward
    ref = hf.model.layers[i](
        hidden_states    = x.clone(),
        attention_mask   = None,
        position_ids     = pos_ids,
        past_key_value   = None,
        output_attentions= False,
        use_cache        = False
    )[0]  # take hidden_states
    out = me.blocks[i](x.clone())
    diff = (ref - out).pow(2).mean().sqrt().item()
    print(" STEP 3 block RMS       =", diff)
