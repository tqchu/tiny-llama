import argparse, torch, math, random
from llm.tinyllama.core.model import TinyLlama
from llm.tinyllama.io.hf_state import load_state
from llm.tinyllama.io.tokenizer import Tok
from transformers import AutoTokenizer, AutoModelForCausalLM

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
ids = torch.tensor([tok.encode(template, add_bos=True, add_eos=True)], device=device)


# — generation loop (top-k sampling) —
with torch.no_grad():
    for _ in range(args.steps):
        next_id = model(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=-1)
        if next_id.item() == tok.eos_id:
            break
print(tok.decode(ids[0].tolist()))


# — SECTION 2: echo debug info —
print("-" * 80)
print("HF vs our tokenizer IDs:")

hf_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
txt = args.prompt
print("HF IDs :", hf_tok.encode(txt, add_special_tokens=True))
print("Our IDs:", Tok(args.tok).encode(txt, add_bos=True))

# — SECTION 3: embedding diff —
print("-" * 80)
print("Embedding max-abs diff:")

hf_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
me = TinyLlama.from_config(args.ckpt + "/config.json")
me.load_state_dict(load_state(args.ckpt), strict=True)
diff = (me.emb.weight - hf_model.model.embed_tokens.weight).abs().max()
print("Embedding max-abs diff:", diff.item())

# — SECTION 4: model‐level logit RMS —
print("-" * 80)
print("Model RMS:")

B, L = 1, 32
prompt_ids = torch.randint(100, 2000, (B, L)).to(device)
with torch.no_grad():
    ref_logits = hf_model(prompt_ids).logits
    my_logits  = model(prompt_ids)
print((ref_logits - my_logits).pow(2).mean().sqrt().item())

# — SECTION 5: RoPE sanity (layer 0) —
print("-" * 80)
print("RoPE max-diff (layer 0):")

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
hf_layer = hf_model.model.layers[0].self_attn
sin_hf, cos_hf = hf_layer.rotary_emb.sin_cos(
    torch.randn(1, 32, 16, hf_layer.rotary_emb.dim), 0
)
sin_my, cos_my = model.blocks[0].attn._build_rope(T=16, device="cpu")
q = torch.randn(1, 32, 16, hf_layer.rotary_emb.dim)
rot_hf, _   = apply_rotary_pos_emb(q.clone(), cos_hf, sin_hf)
rot_my      = model.blocks[0].attn.apply_rope(q.clone(), sin_my, cos_my)
print((rot_hf - rot_my).abs().max().item())

# — SECTION 6: tokenizer sanity —
print("-" * 80)
print("Tokenizer round-trip match?")

base_txt = template
hf_ids   = hf_tok.encode(base_txt, add_special_tokens=False)
my_ids   = Tok(args.tok).encode(base_txt, add_bos=False)
print(hf_ids == my_ids, len(hf_ids), len(my_ids))

# — SECTION 7: final logit check at last token —
print("-" * 80)
print("Logit RMS at last token:")

ids = torch.tensor([my_ids], device=device)
with torch.no_grad():
    lr = hf_model(ids).logits[:, -1, :]
    lm = model(ids)[:, -1, :]
print((lr - lm).pow(2).mean().sqrt().item())
