import argparse, torch, math, random

import numpy as np

from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.hf_state import load_state
from llm.tinyllama.io.tokenizer import Tok
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    default="/Users/chutruong/Academic/GraduationThesis/Projects/ipen-agent/llm/tinyllama/pretrained_1.1b",
    help="Path to model dir (contains config.json & weights)"
)
parser.add_argument(
    "--prompt",
    default="Who is Messi?",
    help="Text prompt"
)
parser.add_argument(
    "--tok",
    default="/Users/chutruong/Academic/GraduationThesis/Projects/ipen-agent/llm/tinyllama/pretrained_1.1b/tokenizer.model",
    help="Path to tokenizer.model"
)
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--topk',  type=int, default=50)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# — load TinyLlama and check state_dict —
model = TinyLlama.from_config(f"{args.ckpt}/config.json").to(device)
with torch.no_grad():
    missing, unexpected = model.load_state_dict(load_state(args.ckpt), strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)
model.eval()

# — tokenizer —
tok = Tok(args.tok)

# — CHAT TEMPLATE (no manual <s>; let Tok.add_bos add it) —
sys_msg = "You are a helpful assistant."
usr_msg = args.prompt.strip()
template = (
    "[INST] <<SYS>>\n"
    f"{sys_msg}\n"
    "<</SYS>>\n\n"
    f"{usr_msg}\n"
    "[/INST]"
)
# encode with a single BOS
input_ids = tok.encode(template, add_bos=True)
    # Prepare model input (as a PyTorch tensor on CPU)
input_tensor = torch.tensor([input_ids], dtype=torch.long)


# — generation loop (top-k sampling) —
generated_tokens = []
    # Generate tokens until max_new_tokens or EOS is encountered
with torch.no_grad():
    # Forward pass for the prompt to get initial logits
    output = model(input_tensor)
    logits = output.logits if hasattr(output, "logits") else output  # get logits tensor
    next_token_logits = logits[0, -1, :]  # logits for the next token after the prompt

    for _ in range(1024):
        # Convert logits to numpy for sampling
        logits_array = next_token_logits.cpu().numpy()
        if args.topk is not None and args.topk > 1:
            # Top-k sampling: select one of the topk tokens based on probability
            vocab_size = logits_array.shape[-1]
            k = args.topk if args.topk < vocab_size else vocab_size
            # Find top k indices and their logits
            sorted_indices = np.argsort(-logits_array)
            topk_indices = sorted_indices[:k]
            topk_logits = logits_array[topk_indices]
            # Compute softmax probabilities over the top k tokens
            max_logit = np.max(topk_logits)
            topk_probs = np.exp(topk_logits - max_logit)
            topk_probs /= np.sum(topk_probs)
            # Randomly sample one of the top k tokens according to the probabilities
            choice = np.random.choice(len(topk_indices), p=topk_probs)
            next_token_id = int(topk_indices[choice])
        else:
            # Greedy decoding (argmax)
            next_token_id = int(np.argmax(logits_array))
        # Stop if EOS token is produced
        if next_token_id == 2:
            break
        # Append the generated token
        generated_tokens.append(next_token_id)
        # Extend the input_ids with the new token and get next logits
        input_ids.append(next_token_id)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        output = model(input_tensor)
        logits = output.logits if hasattr(output, "logits") else output
        next_token_logits = logits[0, -1, :]

# Decode the generated tokens to text
output_text = tok.decode(generated_tokens)
print("-" * 80)
print("Generated response for my model:")
# Print the final generated response
print(output_text)
print("End of reponse.")

print("-" * 80)
print("Debugging HF TinyLlama generation:")

hf_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
txt = template
print("HF IDs :", hf_tok.encode(txt, add_special_tokens=True))

hf_model  = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32, device_map=device)
cfg = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    return_dict_in_generate=True,  # ← required
    output_scores=True,
)

hf_model.eval()
# input_ids = torch.tensor([hf_tok.encode(txt, add_special_tokens=True)], device=hf_model.device)
input_ids = torch.tensor([Tok(args.tok).encode(txt, add_bos=True)], device=hf_model.device)
out = hf_model.generate(
    input_ids= input_ids,)

new_ids = out[0][len(input_ids):]
text = hf_tok.decode(new_ids, skip_special_tokens=True)

print(text)
#
# # — SECTION 2: echo debug info —
# print("-" * 80)
# print("HF vs our tokenizer IDs:")
#
# hf_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# txt = args.prompt
# print("HF IDs :", hf_tok.encode(txt, add_special_tokens=True))
# print("Our IDs:", Tok(args.tok).encode(txt, add_bos=True))
#
# # — SECTION 3: embedding diff —
# print("-" * 80)
# print("Embedding max-abs diff:")
#
# hf_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# me = TinyLlama.from_config(args.ckpt + "/config.json")
# me.load_state_dict(load_state(args.ckpt), strict=True)
# diff = (me.emb.weight - hf_model.model.embed_tokens.weight).abs().max()
# print("Embedding max-abs diff:", diff.item())
#
# # — SECTION 4: model‐level logit RMS —
# print("-" * 80)
# print("Model RMS:")
#
B, L = 1, 32
inputs = hf_tok(txt, return_tensors="pt").to(device)
prompt_ids = inputs.input_ids
with torch.no_grad():
    ref_logits = hf_model(prompt_ids).logits
    my_logits  = model(prompt_ids)
print((ref_logits - my_logits).pow(2).mean().sqrt().item())

print("-" * 80)
print("Debugging TinyLlama embeddings:")
# get HF embeddings
hf_emb = hf_model.get_input_embeddings().weight.detach().cpu()
# get yours
my_emb = model.emb.weight.detach().cpu()

# print the max absolute difference
print("Max embedding diff:", (hf_emb - my_emb).abs().max().item())

hf_sd = hf_model.state_dict()
my_sd = model.state_dict()
max_diff = max(
    (my_sd[k].cpu() - hf_sd[k].cpu()).abs().max().item()
    for k in hf_sd.keys() if k in my_sd
)
print("Max weight diff across all params:", max_diff)

# check parameter dtype
print("hf_model params dtype:", next(hf_model.parameters()).dtype)
print("my model params dtype:", next(model.parameters()).dtype)

# check output dtype
print("ref_logits dtype:", ref_logits.dtype)
print("my_logits dtype:", my_logits.dtype)


# 1) Run the HF model with hidden states
# 1) Run HF once, capturing all hidden states:
hf_model.eval()
with torch.no_grad():
    hf_out = hf_model(
        input_ids=prompt_ids,
        output_hidden_states=True,
        return_dict=True
    )
hf_hiddens = hf_out.hidden_states  # tuple of length n_layers+1

# 2) Run your model block by block, storing each hidden:
my_hiddens = []
x = model.emb(prompt_ids)
my_hiddens.append(x)               # index 0: embeddings
for blk in model.blocks:
    x = blk(x)
    my_hiddens.append(x)           # index i+1: after block i

# 3) Compute RMS at each stage:
for i, (h_hf, h_my) in enumerate(zip(hf_hiddens, my_hiddens)):
    rms = (h_hf - h_my).pow(2).mean().sqrt().item()
    print(f"RMS after {'embeddings' if i==0 else 'block '+str(i-1)}: {rms:.6f}")


# 1) Get HF’s "last_hidden_state" (which includes their final RMSNorm)
hf_model.eval()
with torch.no_grad():
    hf_out = hf_model(
        input_ids=prompt_ids,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False
    )
hf_last = hf_out.hidden_states[-1]              # [B, L, D]

# 2) Run your final norm on your last block output
with torch.no_grad():
    # my_hiddens[-1] is the output after your last block
    my_normed = model.norm(my_hiddens[-1])        # [B, L, D]

# 3) Compute RMS
rms_final_norm = (hf_last - my_normed).pow(2).mean().sqrt().item()
print("RMS after final RMSNorm (before lm_head):", rms_final_norm)

# grab HF’s final RMSNorm layer
hf_norm = hf_model.model.norm  # or wherever the final norm lives

# run it on your pre-norm hidden states
with torch.no_grad():
    hf_normed_my = hf_norm(my_hiddens[-1].to(hf_model.device))

# compare against HF’s true post-norm hidden_state
hf_last = hf_out.hidden_states[-1].to(hf_model.device)
rms_check = (hf_last - hf_normed_my).pow(2).mean().sqrt().item()
print("RMS using HF norm on my features:", rms_check)

# 1) Grab HF’s pre-norm “last block” activation
#    hidden_states[-2] is the output of block 21 (just before the final norm)
hf_pre = hf_out.hidden_states[-2]           # shape [B, L, D]

# 2) Your model’s pre-norm hidden (you stored this as my_hiddens[-1])
my_pre = my_hiddens[-1].to(hf_model.device)  # [B, L, D]

# 3) Compute the RMS difference before norm
rms_pre_norm = (hf_pre - my_pre).pow(2).mean().sqrt().item()
print("RMS before final RMSNorm:", rms_pre_norm)


# assume you already have:

attention_mask = inputs.attention_mask
# if your tokenizer didn’t give you position_ids, build them:
hf_block21 = hf_model.model.layers[21]      # the HF layer you want
captured    = {}

def grab_output(module, inp, out):
    """
    `out` is a tuple: (hidden_states, present_kv, *maybe attn_weights)
    We only need hidden_states.
    """
    captured['hf_hidden21'] = out[0].detach()     # save for later comparison

hook = hf_block21.register_forward_hook(grab_output)

# --- run the full model once (no generate, just a forward pass) ---
hf_model.eval()
with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False,          # so every block runs
                 output_hidden_states=False)

hook.remove()                               # clean up the hook

hf_hidden21 = captured['hf_hidden21']       # tensor [1, 32, 2048]

# now compare with *your* block-21 output
my_hidden21 = my_hiddens[-1].to(hf_hidden21.device)

rms_block21 = (hf_hidden21 - my_hidden21).pow(2).mean().sqrt().item()
print("RMS of block 21 outputs:", rms_block21)


# --- grab HF’s post-attention tensor inside block 21 ---
hf_block21 = hf_model.model.layers[21]
capt = {}

def after_attn_hook(module, inp, out):
    # `out` is the attn result *before* it’s added to residual;
    # the layer then does residual-add → we capture that summed tensor.
    # hidden_states = inp[0]            # residual that enters attn
    attn_out      = out[0]            # attn return
    capt['hf_post_attn'] = attn_out.detach()

hook = hf_block21.self_attn.register_forward_hook(after_attn_hook)

hf_model.eval()
with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False)

hook.remove()

# --- run *your* block-21 up through attention only ---
with torch.no_grad():
    x = model.blocks[20](my_hiddens[-2])      # output of block 20
    x_norm   = model.blocks[21].attn_norm(x)  # your layer-norm before attn
    my_attn  = model.blocks[21].attn(x_norm)  # your self-attn
    my_post_attn = x + my_attn                # add residual

# --- RMS diff just after attention sub-layer ---
rms_attn = (capt['hf_post_attn'] - my_post_attn.to(capt['hf_post_attn'].device))\
           .pow(2).mean().sqrt().item()
print("RMS after attention sub-layer of block 21:", rms_attn)


# --- grab HF’s input to self-attention (output of input_layernorm) ---
hf_block21 = hf_model.model.layers[21]
cap = {}

def ln_hook(module, inp, out):
    cap["hf_ln"] = out.detach()   # out is the normed tensor

ln_handle = hf_block21.input_layernorm.register_forward_hook(ln_hook)

with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False)

ln_handle.remove()

# --- your model’s pre-attention norm on the same residual ---
with torch.no_grad():
    my_ln = model.blocks[21].attn_norm(my_hiddens[-2])   # tensor before QKV

# --- RMS just after the norm (before QKV) ---
rms_ln = (cap["hf_ln"] - my_ln.to(cap["hf_ln"].device)).pow(2).mean().sqrt().item()
print("RMS immediately after input_layernorm:", rms_ln)

# --- 1) capture HF q_proj output ---------------------------------
hf_q = {}

def q_hook(module, inp, out):
    hf_q["q"] = out.detach()      # out is [batch, seq_len, n_heads*head_dim]

q_handle = hf_block21.self_attn.q_proj.register_forward_hook(q_hook)

with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False)

q_handle.remove()

# --- 2) run your q_proj on the same pre-norm tensor ---------------
with torch.no_grad():
    my_ln   = model.blocks[21].attn_norm(my_hiddens[-2])
    my_q    = model.blocks[21].attn.q_proj(my_ln)     # your q projection

# --- 3) RMS between the two --------------------------------------
rms_q = (hf_q["q"] - my_q.to(hf_q["q"].device)).pow(2).mean().sqrt().item()
print("RMS of q_proj outputs:", rms_q)

# --- 1) capture HF q_proj output ---------------------------------
hf_k = {}

def k_hook(module, inp, out):
    hf_k["k"] = out.detach()      # out is [batch, seq_len, n_heads*head_dim]

k_handle = hf_block21.self_attn.k_proj.register_forward_hook(k_hook)

with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False)

k_handle.remove()

# --- 2) run your q_proj on the same pre-norm tensor ---------------
with torch.no_grad():
    my_ln   = model.blocks[21].attn_norm(my_hiddens[-2])
    my_k    = model.blocks[21].attn.k_proj(my_ln)     # your q projection

# --- 3) RMS between the two --------------------------------------
rms_k = (hf_k["k"] - my_k.to(hf_k["k"].device)).pow(2).mean().sqrt().item()
print("RMS of k_proj outputs:", rms_k)


hf_v = {}

def v_hook(module, inp, out):
    hf_v["v"] = out.detach()      # out is [batch, seq_len, n_heads*head_dim]

v_handle = hf_block21.self_attn.v_proj.register_forward_hook(v_hook)

with torch.no_grad():
    _ = hf_model(input_ids=prompt_ids,
                 attention_mask=attention_mask,
                 use_cache=False)

v_handle.remove()

# --- 2) run your q_proj on the same pre-norm tensor ---------------
with torch.no_grad():
    my_ln   = model.blocks[21].attn_norm(my_hiddens[-2])
    my_v    = model.blocks[21].attn.v_proj(my_ln)     # your q projection

# --- 3) RMS between the two --------------------------------------
rms_v = (hf_v["v"] - my_v.to(hf_v["v"].device)).pow(2).mean().sqrt().item()
print("RMS of v_proj outputs:", rms_v)

print("HF shapes  — q,k,v:")
print(" q_proj:", hf_block21.self_attn.q_proj.weight.shape)
print(" k_proj:", hf_block21.self_attn.k_proj.weight.shape)
print(" v_proj:", hf_block21.self_attn.v_proj.weight.shape)

print("\nYour shapes — q,k,v:")
print(" q_proj:", model.blocks[21].attn.q_proj.weight.shape)
print(" k_proj:", model.blocks[21].attn.k_proj.weight.shape)
print(" v_proj:", model.blocks[21].attn.v_proj.weight.shape)

print("HF eps for block-21 input_layernorm :",
      hf_block21.input_layernorm.variance_epsilon)

print("Your eps for block-21 attn_norm      :",
      model.blocks[21].attn_norm.eps)

# --- 1) capture HF's rotary-embedded q/k --------------------------
hf_rot = {}
def rope_hook(mod, inp, out):
    # mod is LlamaRotaryEmbedding, out is a tuple (q_rot, k_rot)
    hf_rot["q_rot"], hf_rot["k_rot"] = [t.detach() for t in out]

rope_handle = hf_block21.self_attn.rotary_emb.register_forward_hook(rope_hook)

with torch.no_grad():
    _ = hf_model(prompt_ids, attention_mask=attention_mask, use_cache=False)

rope_handle.remove()

# --- 2) run your rotary embedding on q/k --------------------------
# inside your diagnostic snippet *before* calling rope:
with torch.no_grad():
    my_ln = model.blocks[21].attn_norm(my_hiddens[-2])

    B, T, _ = my_ln.shape
    H       = model.blocks[21].attn.cfg.n_heads
    HK      = model.blocks[21].attn.cfg.n_kv_heads
    D       = model.blocks[21].attn.d_head            # head-di

    # Project and RESHAPE first
    my_q = model.blocks[21].attn.q_proj(my_ln).view(B, T, H, D).transpose(1, 2)
    my_k = model.blocks[21].attn.k_proj(my_ln).view(B, T, HK, D).transpose(1, 2)

    # now q, k are 4-D → safe to call rope
    my_q_rot, my_k_rot = model.blocks[21].attn.rope(my_q, my_k)


# --- 3) measure RMS right after rope ------------------------------
rms_q_rot = (hf_rot["q_rot"] - my_q_rot.to(hf_rot["q_rot"].device)).pow(2).mean().sqrt().item()
rms_k_rot = (hf_rot["k_rot"] - my_k_rot.to(hf_rot["k_rot"].device)).pow(2).mean().sqrt().item()

print("RMS after rope — q :", rms_q_rot)
print("RMS after rope — k :", rms_k_rot)

# Sequence length & rotary dim that block-21 actually uses
T  = my_hiddens[-2].size(1)                        # 32
Dr = hf_block21.self_attn.rotary_emb.dim           # usually 128 in TinyLlama

# 1) Grab HF’s cached tables for the first T positions
hf_sin = hf_block21.self_attn.rotary_emb.sin_cached[:T, :Dr].detach()
hf_cos = hf_block21.self_attn.rotary_emb.cos_cached[:T, :Dr].detach()

# 2) Build YOUR tables the same way
sin_my, cos_my = model.blocks[21].attn.build_rope(T, hf_sin.device)
sin_my, cos_my = sin_my[0, :T, :Dr], cos_my[0, :T, :Dr]   # shape match HF

print("HF rotary dim :", hf_block21.self_attn.rotary_emb.dim)   # e.g. 32 or 64
print("Head dim (D)  :", model.blocks[21].attn.d_head)          # 64 in your cfg

print("HF sin shape :", hf_block21.self_attn.rotary_emb.sin_cached[:T].shape)
print("my  sin shape:", model.blocks[21].attn.build_rope(T, hf_sin.device)[0][0,:T].shape)


# 3) RMS difference
rms_sin = (hf_sin - sin_my).pow(2).mean().sqrt().item()
rms_cos = (hf_cos - cos_my).pow(2).mean().sqrt().item()
print("RMS between HF & your sin table :", rms_sin)
print("RMS between HF & your cos table :", rms_cos)

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# assume you have:
#   prompt_ids, attention_mask, hf_model, model, my_hiddens, hf_block21 already defined
# and my_hiddens[-2] is the pre‐attention hidden of block 21
with torch.no_grad():
    # 1) your pre-rope Q/K in (B, H, T, D)
    q4d = my_q    # shape [1,32,33,64]
    k4d = my_k    # shape [1, 4,33,64]

    B, H, T, D = q4d.shape

    # 2) build a (B, T) position_ids tensor (0…T-1) matching your batch
    pos_ids = torch.arange(T, device=q4d.device).unsqueeze(0).repeat(B, 1)  # [1,33]

    # 3) grab HF’s full-headed sin / cos (using the new forward API)
    cos_hf, sin_hf = hf_block21.self_attn.rotary_emb(
        q4d.to(hf_model.device),
        pos_ids
    )

    # 4) apply their official RoPE to both Q/K
    #    Note unsqueeze_dim=1 so (1,1,33,64) broadcasts against (1,32,33,64)
    hf_q_rot, hf_k_rot = apply_rotary_pos_emb(
        q4d.to(hf_model.device),
        k4d.to(hf_model.device),
        cos=cos_hf,
        sin=sin_hf,
        unsqueeze_dim=1
    )

    # 5) compare to yours
    my_q_rot, my_k_rot = model.blocks[21].attn.rope(q4d, k4d)
    rms_q = (hf_q_rot - my_q_rot.to(hf_q_rot.device)).pow(2).mean().sqrt().item()
    rms_k = (hf_k_rot - my_k_rot.to(hf_k_rot.device)).pow(2).mean().sqrt().item()
    print("RMS HF vs my rope → q:", rms_q, " k:", rms_k)

    hf_q0 = hf_q_rot[0, 0]  # [T,D]
    hf_k0 = hf_k_rot[0, 0]  # [T,D]
    my_q0 = my_q_rot[0, 0]
    my_k0 = my_k_rot[0, 0]

    # 1) raw scores (before scaling/mask)
    scores_hf = hf_q0 @ hf_k0.transpose(-1, -2)  # [T,T]
    scores_my = my_q0 @ my_k0.transpose(-1, -2)

    # 2) scale
    scaling = 1.0 / math.sqrt(hf_q0.size(-1))
    scores_hf = scores_hf * scaling
    scores_my = scores_my * scaling

    print("RMS of raw scores (head 0):", torch.dist(scores_hf, scores_my).item())

    # (Optionally, print a small slice)
    print("scores_hf[0,:5]:", scores_hf[0, :5])
    print("scores_my[0,:5]:", scores_my[0, :5])

    masked_hf = scores_hf.masked_fill(attention_mask == 0, float("-inf"))
    masked_my = scores_my.masked_fill(attention_mask == 0, float("-inf"))

    import torch.nn.functional as F
    probs_hf = F.softmax(masked_hf, dim=-1)
    probs_my = F.softmax(masked_my, dim=-1)

    print("RMS of softmax probs:", torch.dist(probs_hf, probs_my).item())
    print("probs_hf[0,:5]:", probs_hf[0, :5])
    print("probs_my[0,:5]:", probs_my[0, :5])

    hidden = my_hiddens[-2]  # shape [B, T, C]

    # 2) HF’s value projection → (B, T, C)
    hidden_hf = hidden.to(hf_model.device)
    v_hf = hf_block21.self_attn.v_proj(hidden_hf)

    # 3) reshape + permute into (B, H, T, D)
    B, T, C = v_hf.shape
    heads = hf_block21.self_attn.num_heads
    D = C // heads
    hf_v_rot = v_hf.view(B, T, heads, D).permute(0, 2, 1, 3)

    # — now do the same with your TinyLlama block21 —
    v_my = model.blocks[21].attn.v_proj(hidden)  # stays on your device
    v_my4 = v_my.view(B, T, heads, D).permute(0, 2, 1, 3)
    my_v_rot = v_my4

    hf_q0 = hf_q_rot[0, 0]  # [T, D]
    hf_k0 = hf_k_rot[0, 0]  # [T, D]
    hf_v0 = hf_v_rot[0, 0]  # [T, D]

    my_q0 = my_q_rot[0, 0]
    my_k0 = my_k_rot[0, 0]
    my_v0 = my_v_rot[0, 0]

    # 2) raw scores and scaling (we know these match)
    scores_hf = hf_q0 @ hf_k0.transpose(-1, -2) * (1.0 / math.sqrt(hf_q0.size(-1)))
    scores_my = my_q0 @ my_k0.transpose(-1, -2) * (1.0 / math.sqrt(my_q0.size(-1)))

    # 3) causal mask → softmax (we know these match too)
    mask = attention_mask[0, 0]  # [T, T], 1s and 0s
    scores_hf = scores_hf.masked_fill(mask == 0, float("-inf"))
    scores_my = scores_my.masked_fill(mask == 0, float("-inf"))

    probs_hf = F.softmax(scores_hf, dim=-1)  # [T, T]
    probs_my = F.softmax(scores_my, dim=-1)

    # 4) compute the context vectors
    ctx_hf = probs_hf @ hf_v0  # [T, D]
    ctx_my = probs_my @ my_v0

    print("RMS of context vectors:", torch.dist(ctx_hf, ctx_my).item())
    print("ctx_hf[0,:5]:", ctx_hf[0, :5])
    print("ctx_my[0,:5]:", ctx_my[0, :5])

    B, H, T, D = hf_v_rot.shape
    C = H * D

    # 1) rebuild full context for all heads
    ctx_hf_all = torch.einsum("bhtm,bhmd->bhtd", probs_all, hf_v_rot)  # [1,H,T,D]
    ctx_my_all = torch.einsum("bhtm,bhmd->bhtd", probs_all, my_v_rot)  # [1,H,T,D]

    # 2) flatten heads exactly as HF does: (B,H,T,D) → (B,T,H,D) → (B,T,C)
    hf_flat = ctx_hf_all.permute(0, 2, 1, 3).reshape(B, T, C)
    my_flat = ctx_my_all.permute(0, 2, 1, 3).reshape(B, T, C)

    # 3) apply each model’s o_proj
    hf_out = hf_block21.self_attn.o_proj(hf_flat)
    my_out = model.blocks[21].attn.o_proj(my_flat)

    # 4) compare
    rms_out = (hf_out - my_out).pow(2).mean().sqrt().item()
    print("RMS of o_proj output:", rms_out)
    print("First 5 HF-out:", hf_out[0, :5])
    print("First 5 my-out:", my_out[0, :5])


# pick batch=0, head=0, pos=0…T-1, dim=0…D-1
q_slice = q4d[0:1, 0:1, :, :].clone().to(hf_model.device)  # shape [1,1,T,D]
k_slice = k4d[0:1, 0:1, :, :].clone().to(hf_model.device)

cos, sin = hf_block21.self_attn.rotary_emb(q_slice, pos_ids)
hf_qs, hf_ks = apply_rotary_pos_emb(q_slice, k_slice, cos=cos, sin=sin)
my_qs, my_ks   = model.blocks[21].attn.rope(q_slice, k_slice)

print("RMS Q-slice:", torch.dist(hf_qs, my_qs.to(hf_qs.device)))
print("RMS K-slice:", torch.dist(hf_ks, my_ks.to(hf_ks.device)))
print("First 5 elements HF-Q:", hf_qs.flatten()[:5])
print("First 5 elements My-Q:", my_qs.flatten()[:5])


import torch
from transformers.models.llama.modeling_llama import rotate_half

# 1) a simple “head” of size D=8 so we can eyeball it
x = torch.arange(8).float().unsqueeze(0)  # shape [1,8]

# 2) HF’s split-half rotate
hf_rot = rotate_half(x)

# 3) your adjacent-pair version
x_even = x[..., ::2]    # [0,2,4,6]
x_odd  = x[..., 1::2]   # [1,3,5,7]
my_rot = torch.cat((-x_odd, x_even), dim=-1)

print("x:      ", x.flatten().tolist())
print("hf_rot: ", hf_rot.flatten().tolist())
print("my_rot: ", my_rot.flatten().tolist())


#
# # — SECTION 5: RoPE sanity (layer 0) —
# print("-" * 80)
# print("RoPE max-diff (layer 0):")
#
# from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# hf_layer = hf_model.model.layers[0].self_attn
# sin_hf, cos_hf = hf_layer.rotary_emb.sin_cos(
#     torch.randn(1, 32, 16, hf_layer.rotary_emb.dim), 0
# )
# sin_my, cos_my = model.blocks[0].attn._build_rope(T=16, device="cpu")
# q = torch.randn(1, 32, 16, hf_layer.rotary_emb.dim)
# rot_hf, _   = apply_rotary_pos_emb(q.clone(), cos_hf, sin_hf)
# rot_my      = model.blocks[0].attn.apply_rope(q.clone(), sin_my, cos_my)
# print((rot_hf - rot_my).abs().max().item())
#
# # — SECTION 6: tokenizer sanity —
# print("-" * 80)
# print("Tokenizer round-trip match?")
#
# base_txt = template
# hf_ids   = hf_tok.encode(base_txt, add_special_tokens=False)
# my_ids   = Tok(args.tok).encode(base_txt, add_bos=False)
# print(hf_ids == my_ids, len(hf_ids), len(my_ids))
#
# # — SECTION 7: final logit check at last token —
# print("-" * 80)
# print("Logit RMS at last token:")
#
# ids = torch.tensor([my_ids], device=device)
# with torch.no_grad():
#     lr = hf_model(ids).logits[:, -1, :]
#     lm = model(ids)[:, -1, :]
# print((lr - lm).pow(2).mean().sqrt().item())
