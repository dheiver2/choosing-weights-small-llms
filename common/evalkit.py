"""Avaliação de modelos pequenos em múltipla escolha (log-verossimilhança) e perplexidade.

Convenções (estilo lm-eval): arc_easy usa acc_norm por caracteres, arc_challenge acc_norm, hellaswag acc_norm,
winogrande acc com pontuação parcial. Cada avaliação devolve o vetor de acerto POR ITEM (0/1) para permitir
bootstrap pareado. Divisões val/teste fixas e disjuntas (seed 0), criadas aqui e nunca alteradas.
"""
import os, re, json, hashlib
import numpy as np, torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.bfloat16
N_VAL, N_TEST = 300, 1000          # itens por tarefa: seleção (val) e relatório (teste)
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def _norm_hs(t):
    t = t.strip().replace(" [title]", ". "); t = re.sub(r"\[.*?\]", "", t); return t.replace("  ", " ")


def _items(task):
    """Lista de itens {ctx:[...], cont:[...], gold:int, norm:bool} em ordem fixa; val = primeiros N_VAL, teste = próximos N_TEST."""
    d = load_from_disk(os.path.join(ROOT, "data", task))
    rng = np.random.default_rng(0)
    if task in ("arc_easy", "arc_challenge"):
        pool_val, pool_test = d["validation"], d["test"]
        out_v = [dict(ctx=[f"Question: {x['question']}\nAnswer:"] * len(x["choices"]["text"]),
                      cont=[" " + c for c in x["choices"]["text"]], gold=x["choices"]["label"].index(x["answerKey"]), norm=True) for x in pool_val]
        out_t = [dict(ctx=[f"Question: {x['question']}\nAnswer:"] * len(x["choices"]["text"]),
                      cont=[" " + c for c in x["choices"]["text"]], gold=x["choices"]["label"].index(x["answerKey"]), norm=True) for x in pool_test]
        iv = rng.permutation(len(out_v))[:N_VAL]; it = rng.permutation(len(out_t))[:N_TEST]
        return [out_v[i] for i in iv], [out_t[i] for i in it]
    if task == "hellaswag":
        pool = d["validation"]
        allit = []
        for x in pool:
            c = _norm_hs(x["activity_label"] + ": " + x["ctx_a"] + " " + x["ctx_b"].capitalize())
            allit.append(dict(ctx=[c] * 4, cont=[" " + _norm_hs(e) for e in x["endings"]], gold=int(x["label"]), norm=True))
        idx = rng.permutation(len(allit)); return [allit[i] for i in idx[:N_VAL]], [allit[i] for i in idx[N_VAL:N_VAL + N_TEST]]
    if task == "winogrande":
        pool = d["validation"]; allit = []
        for x in pool:
            i = x["sentence"].index("_"); ans = [x["option1"], x["option2"]]
            allit.append(dict(ctx=[x["sentence"][:i] + a for a in ans], cont=[" " + x["sentence"][i + 1:].strip()] * 2, gold=int(x["answer"]) - 1, norm=False))
        idx = rng.permutation(len(allit)); return [allit[i] for i in idx[:N_VAL]], [allit[i] for i in idx[N_VAL:N_VAL + N_TEST]]
    raise ValueError(task)


_ITEM_CACHE = {}
def items(task, split):
    if task not in _ITEM_CACHE: _ITEM_CACHE[task] = _items(task)
    return _ITEM_CACHE[task][0 if split == "val" else 1]


def load(path, dtype=DTYPE):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(DEV).eval()
    return model, tok


@torch.no_grad()
def _loglik(model, tok, pairs, bs=16):
    """pairs: lista de (ctx, cont). Devolve (soma_logprob_da_continuação, n_chars)."""
    enc = []
    for c, k in pairs:
        ci = tok(c, add_special_tokens=False)["input_ids"]; ki = tok(c + k, add_special_tokens=False)["input_ids"]
        n = len(ci)
        while n > 0 and ki[:n] != ci[:n]: n -= 1      # fronteira de token robusta
        enc.append((ki, n))
    order = sorted(range(len(enc)), key=lambda i: -len(enc[i][0])); res = [0.0] * len(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; L = max(len(enc[i][0]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ki = enc[i][0]; ids[r, :len(ki)] = torch.tensor(ki); att[r, :len(ki)] = 1
        lg = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits   # bf16; só fatiamos as posições da continuação
        for r, i in enumerate(idx):
            ki, n = enc[i]; a = max(n - 1, 0)
            lp = torch.log_softmax(lg[r, a:len(ki) - 1].float(), -1)
            tgt = torch.tensor(ki[a + 1:], device=DEV)
            res[i] = float(lp.gather(-1, tgt.unsqueeze(-1)).sum())
        del lg
    return res


def eval_task(model, tok, task, split):
    """Vetor 0/1 de acerto por item."""
    its = items(task, split); pairs = []; spans = []
    for it in its:
        spans.append((len(pairs), len(it["cont"])))
        pairs += list(zip(it["ctx"], it["cont"]))
    ll = _loglik(model, tok, pairs); out = np.zeros(len(its), dtype=np.int8)
    for j, (it, (s, n)) in enumerate(zip(its, spans)):
        sc = np.array(ll[s:s + n])
        if it["norm"]: sc = sc / np.array([max(len(c), 1) for c in it["cont"]])
        out[j] = int(np.argmax(sc) == it["gold"])
    return out


def eval_suite(model, tok, tasks=TASKS, split="val"):
    return {t: eval_task(model, tok, t, split) for t in tasks}


def macro(res):
    return float(np.mean([r.mean() for r in res.values()]))


@torch.no_grad()
def wikitext_nll(model, tok, n_chunks=40, L=512):
    """NLL médio por token em n_chunks blocos fixos do teste do WikiText-2 (menor = melhor)."""
    d = load_from_disk(os.path.join(ROOT, "data", "wikitext2"))["test"]
    ids = tok("\n\n".join(d["text"]), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    out = []
    for k in range(n_chunks):
        x = ids[k * L:(k + 1) * L + 1]
        if len(x) < L + 1: break
        lg = model(input_ids=x[:-1].unsqueeze(0).to(DEV)).logits.float()[0]
        out.append(float(torch.nn.functional.cross_entropy(lg, x[1:].to(DEV))))
    return np.array(out)
