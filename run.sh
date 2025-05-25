pip install torch sentencepiece
python -m tinyllama.runtime.generate \
  --ckpt ./TinyLlama-1.1B-Chat-v1.0 \
  --tok ./TinyLlama-1.1B-Chat-v1.0/tokenizer.model \
  --prompt "List three ways to fingerprint an open SSH port."