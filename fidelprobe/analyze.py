"""Scores the predictions and writes summary json, a markdown table and plots."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import decompose, error_breakdown, is_ethiopic, is_hangul, substitution_type

SCRIPT_TEST = {"eth": is_ethiopic, "hangul": is_hangul}
LABEL = {"eth": "Ethiopic (Amharic)", "hangul": "Hangul (Korean control)"}
ERR_KEYS = ["vowel_form", "consonant", "both", "script_escape", "fragment", "other", "del", "ins"]


def extract(item, pred):
    t = item["answer_type"]
    if t == "digit":
        m = re.search(r"[1-7]", pred)
        return m.group(0) if m else ""
    if t == "yesno":
        m = re.search(r"\b(yes|no)\b", pred.lower())
        return m.group(1) if m else ""
    if t == "char":
        # take the first char in the expected script, otherwise whatever came first
        same = [c for c in pred if SCRIPT_TEST[item["script"]](c) or c in "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"]
        if same:
            return same[0]
        rest = [c for c in pred if not c.isspace()]
        return rest[0] if rest else ""
    return " ".join(pred.split())


def score(items, preds):
    rows = []
    for it in items:
        if it["id"] not in preds:
            continue
        raw = preds[it["id"]]
        ans = extract(it, raw)
        row = {**it, "raw": raw, "answer": ans, "correct": ans == it["gold"]}
        if it["task"] in ("compose", "transform", "ocr_char") or (
                it["task"] == "decompose_consonant" and it["script"] == "eth"):
            row["err_type"] = substitution_type(it["gold"], ans) if ans else "empty"
        if it["task"] in ("ocr_word", "ocr_line"):
            g = " ".join(it["gold"].split())
            row["breakdown"] = error_breakdown(g, ans)
        rows.append(row)
    return rows


def summarize(rows):
    S = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["script"])].append(r)
    for (task, script), rs in sorted(by.items()):
        d = {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs)}
        if "err_type" in rs[0]:
            errs = Counter(r["err_type"] for r in rs if not r["correct"])
            d["errors"] = dict(errs)
            wrong = sum(errs.values())
            d["vowel_form_share_of_errors"] = errs.get("vowel_form", 0) / wrong if wrong else 0.0
        if task == "minimal_pair":
            for cond in ["identical", "vowel", "consonant"]:
                cs = [r for r in rs if r["condition"] == cond]
                if cs:
                    d[f"acc_{cond}"] = sum(r["correct"] for r in cs) / len(cs)
        if "breakdown" in rs[0]:
            tot = Counter()
            for r in rs:
                tot.update(r["breakdown"])
            edits = sum(tot[k] for k in ERR_KEYS)
            subs = sum(tot[k] for k in ["vowel_form", "consonant", "both", "script_escape", "fragment", "other"])
            d["cer"] = edits / tot["ref_len"]
            d["edits"] = {k: tot[k] for k in ERR_KEYS}
            d["vowel_form_share_of_substitutions"] = tot["vowel_form"] / subs if subs else 0.0
            d["exact_match"] = d.pop("accuracy")
        if task in ("ocr_char", "identify_order", "transform", "compose"):
            key = "target_order" if task in ("compose", "transform") else "order"
            per = defaultdict(list)
            for r in rs:
                per[r[key]].append(r["correct"])
            d["acc_by_order"] = {k: sum(v) / len(v) for k, v in sorted(per.items())}
        S[task][script] = d
    return S


def fig_task_bars(S, path):
    tasks = [t for t in ["ocr_char", "decompose_consonant", "identify_order", "compose", "transform",
                         "minimal_pair"] if t in S]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.38
    for k, (script, color) in enumerate([("eth", "#c2410c"), ("hangul", "#1d4ed8")]):
        vals = [S[t].get(script, {}).get("accuracy", 0) for t in tasks]
        bars = ax.bar([i + (k - 0.5) * w for i in range(len(tasks))], vals, w, label=LABEL[script],
                      color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    chance = {"identify_order": 1 / 7, "minimal_pair": 0.5}
    for i, t in enumerate(tasks):
        if t in chance:
            ax.hlines(chance[t], i - 0.45, i + 0.45, colors="gray", linestyles=":", lw=1)
    ax.set_xticks(range(len(tasks)), [t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_order_confusion(rows, path):
    """Vowel-order confusion for single-character OCR, only where the consonant was right."""
    M = [[0] * 7 for _ in range(7)]
    for r in rows:
        if r["task"] != "ocr_char" or r["script"] != "eth" or not r["answer"]:
            continue
        g, a = decompose(r["gold"]), decompose(r["answer"])
        if a and a[0] == "eth" and a[1] == g[1] and a[2] < 7:
            M[g[2]][a[2]] += 1
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(M, cmap="Oranges")
    for i in range(7):
        for j in range(7):
            if M[i][j]:
                ax.text(j, i, M[i][j], ha="center", va="center", fontsize=8,
                        color="white" if M[i][j] > max(map(max, M)) / 2 else "black")
    lab = ["1 ä", "2 u", "3 i", "4 a", "5 e", "6 ə", "7 o"]
    ax.set_xticks(range(7), lab, fontsize=8)
    ax.set_yticks(range(7), lab, fontsize=8)
    ax.set_xlabel("predicted order (correct consonant)")
    ax.set_ylabel("true order")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_error_mix(S, path):
    cats = [("vowel_form", "#c2410c"), ("consonant", "#1d4ed8"), ("both", "#7c3aed"),
            ("script_escape", "#6b7280"), ("del", "#a3a3a3"), ("ins", "#d4d4d4")]
    bars = [(f"{t.split('_')[1]}\n{s}", S[t][s]) for t in ["ocr_word", "ocr_line"] if t in S
            for s in ["eth", "hangul"] if s in S[t]]
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, d) in enumerate(bars):
        left = 0
        for c, col in cats:
            v = d["edits"].get(c, 0) / max(1, sum(d["edits"].values())) * d["cer"]
            ax.barh(i, v, left=left, color=col, label=c if i == 0 else None)
            left += v
        ax.text(left + 0.005, i, f"CER {d['cer']:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("character error rate, split by edit type")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}%"


def results_md(S, tag, tok=None):
    L = [f"### Results: `{tag.replace('__', '/')}`", ""]
    L += ["**Headline: Ethiopic vs the matched Hangul control (accuracy)**", "",
          "| Task | Ethiopic | Hangul | What it tests |", "|---|---|---|---|"]
    desc = {"ocr_char": "read one rendered character",
            "decompose_consonant": "name the base consonant of a character",
            "identify_order": "name the vowel order 1-7 (chance 14%)",
            "compose": "consonant + order -> character",
            "transform": "same consonant, new vowel order",
            "minimal_pair": "same/different spelling (chance 50%)"}
    for t, dsc in desc.items():
        if t in S:
            e, h = S[t].get("eth", {}), S[t].get("hangul", {})
            L.append(f"| {t} | {pct(e.get('accuracy', 0))} | {pct(h.get('accuracy', 0))} | {dsc} |")
    if "minimal_pair" in S:
        L += ["", "**Minimal pairs: accuracy by the kind of difference**", "",
              "| | identical | one vowel form changed | one consonant changed |", "|---|---|---|---|"]
        for s in ["eth", "hangul"]:
            d = S["minimal_pair"].get(s, {})
            L.append(f"| {LABEL[s]} | {pct(d.get('acc_identical', 0))} | {pct(d.get('acc_vowel', 0))} "
                     f"| {pct(d.get('acc_consonant', 0))} |")
    ocr = [(t, s) for t in ["ocr_word", "ocr_line"] if t in S for s in ["eth", "hangul"] if s in S[t]]
    if ocr:
        L += ["", "**Rendered-text OCR (character error rate, and what the errors are)**", "",
              "| Set | CER | exact match | vowel-form subs | consonant subs | other-script subs |"
              " deletions | insertions |", "|---|---|---|---|---|---|---|---|"]
        for t, s in ocr:
            d = S[t][s]
            e = d["edits"]
            L.append(f"| {t} / {s} | {pct(d['cer'])} | {pct(d['exact_match'])} | {e['vowel_form']} "
                     f"| {e['consonant']} | {e['script_escape']} | {e['del']} | {e['ins']} |")
    err_tasks = [t for t in ["ocr_char", "compose", "transform"] if t in S and "eth" in S[t]]
    if err_tasks:
        L += ["", "**Ethiopic single-character errors by type**", "",
              "| Task | vowel form (right consonant) | consonant | both | other script | n wrong |",
              "|---|---|---|---|---|---|"]
        for t in err_tasks:
            e = S[t]["eth"].get("errors", {})
            L.append(f"| {t} | {e.get('vowel_form', 0)} | {e.get('consonant', 0)} | {e.get('both', 0)} "
                     f"| {e.get('script_escape', 0)} | {sum(e.values())} |")
    if "ocr_char" in S and "eth" in S["ocr_char"]:
        bo = S["ocr_char"]["eth"]["acc_by_order"]
        L += ["", "**Ethiopic single-character OCR accuracy by vowel order**", "",
              "| " + " | ".join(f"{k} ({o})" for k, o in zip(range(1, 8), "ä u i a e ə o".split())) + " |",
              "|" + "---|" * 7, "| " + " | ".join(pct(bo.get(str(k), bo.get(k, 0))) for k in range(1, 8)) + " |"]
    if tok:
        L += ["", "**Tokenizer view of the Fidel grid**", "",
              "| | single-token chars | mean tokens / char | consonants whose 7 forms share a token |",
              "|---|---|---|---|"]
        for s in ["eth", "hangul"]:
            d = tok[s]
            L.append(f"| {LABEL[s]} | {d['single_token_chars']}/{d['n_chars']} | "
                     f"{d['mean_tokens_per_char']:.2f} | {d['consonants_with_shared_token_across_forms']} |")
        ln = tok["lines"]
        L.append("")
        L.append("Tokens per character on sentences: " + ", ".join(
            f"{k} {v['tokens_per_char']:.2f}" for k, v in ln.items()) + ".")
    L += ["", f"![task accuracy](results/fig_tasks_{tag}.png)",
          f"![vowel-order confusion](results/fig_order_confusion_{tag}.png)",
          f"![OCR errors](results/fig_ocr_errors_{tag}.png)"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(Path(args.data) / "probes.jsonl", encoding="utf-8")]
    res = Path(args.results)
    for pf in sorted(res.glob("predictions_*.jsonl")):
        tag = pf.stem.replace("predictions_", "")
        preds = {}
        for l in open(pf, encoding="utf-8"):
            p = json.loads(l)
            preds[p["id"]] = p["pred"]
        rows = score(items, preds)
        S = summarize(rows)
        (res / f"summary_{tag}.json").write_text(json.dumps(S, indent=2, ensure_ascii=False))
        with open(res / f"scored_{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                r.pop("prompt", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        fig_task_bars(S, res / f"fig_tasks_{tag}.png")
        fig_order_confusion(rows, res / f"fig_order_confusion_{tag}.png")
        fig_error_mix(S, res / f"fig_ocr_errors_{tag}.png")
        tok_path = res / f"tokenizer_{tag}.json"
        tok = json.loads(tok_path.read_text()) if tok_path.exists() else None
        (res / f"RESULTS_{tag}.md").write_text(results_md(S, tag, tok), encoding="utf-8")
        print(tag)
        print(json.dumps(S, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
