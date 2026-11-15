"""Baixa modelos (revisões fixas) e datasets pequenos para ./models e ./data. Registra os commits em MANIFEST.json."""
import json, os, sys
from huggingface_hub import snapshot_download, HfApi
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
M = os.path.join(ROOT, "models"); D = os.path.join(ROOT, "data")
MODELOS = {
    "qwen25-05b": ("Qwen/Qwen2.5-0.5B", None),
    "qwen25-05b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", None),
    "qwen25-coder-05b-instruct": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", None),
    "smollm2-360m": ("HuggingFaceTB/SmolLM2-360M", None),
    "smollm2-360m-instruct": ("HuggingFaceTB/SmolLM2-360M-Instruct", None),
}
PYTHIA_STEPS = [143000, 140000, 130000, 120000, 110000, 100000]
api = HfApi(); manifest = {"models": {}, "datasets": {}}
def baixar(nome, repo, rev):
    dst = os.path.join(M, nome)
    p = snapshot_download(repo, revision=rev, local_dir=dst,
                          allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.*", "tokenizer*", "*.model"])
    sha = api.model_info(repo, revision=rev).sha
    manifest["models"][nome] = {"repo": repo, "revision": rev or "main", "sha": sha}
    print("ok", nome, sha[:8], flush=True)
if "modelos" in sys.argv:
    for n, (r, rev) in MODELOS.items(): baixar(n, r, rev)
if "pythia" in sys.argv:
    for s in PYTHIA_STEPS: baixar(f"pythia160m-step{s}", "EleutherAI/pythia-160m", f"step{s}")
if "dados" in sys.argv:
    from datasets import load_dataset
    specs = [("arc_easy", ("allenai/ai2_arc", "ARC-Easy")), ("arc_challenge", ("allenai/ai2_arc", "ARC-Challenge")),
             ("hellaswag", ("Rowan/hellaswag", None)), ("winogrande", ("allenai/winogrande", "winogrande_xl")),
             ("gsm8k", ("openai/gsm8k", "main")), ("wikitext2", ("Salesforce/wikitext", "wikitext-2-raw-v1"))]
    for n, (r, c) in specs:
        try:
            ds = load_dataset(r, c) if c else load_dataset(r)
            ds.save_to_disk(os.path.join(D, n)); manifest["datasets"][n] = {"repo": r, "config": c, "splits": {k: len(v) for k, v in ds.items()}}
            print("ok", n, {k: len(v) for k, v in ds.items()}, flush=True)
        except Exception as e: print("FALHOU", n, repr(e)[:200], flush=True)
mp = os.path.join(ROOT, "MANIFEST.json")
old = json.load(open(mp)) if os.path.exists(mp) else {"models": {}, "datasets": {}}
old["models"].update(manifest["models"]); old["datasets"].update(manifest["datasets"]); json.dump(old, open(mp, "w"), indent=2)
