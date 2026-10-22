"""Checks how the tokenizer splits Fidel characters.

If the seven forms of a consonant never share a token, the model has no
direct way to tell from its input that ቡ and ቦ are the same consonant.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from build_dataset import AMHARIC_LINES, KOREAN_LINES
from scripts import eth_core_chars, hangul_core_chars

ENGLISH_LINES = [  # translations of the first few Amharic lines
    "Hello, how are you?", "Addis Ababa is the capital city of Ethiopia.", "I love drinking coffee.",
    "The children went to school.", "It will rain today.", "My mother baked injera.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    report = {"model": args.model}
    for script, chars in [("eth", eth_core_chars()), ("hangul", hangul_core_chars())]:
        ids = {ch: enc(ch) for ch, _, _ in chars}
        by_base = defaultdict(list)
        for ch, base, _ in chars:
            by_base[base].append(ids[ch])
        # does every form of this consonant share at least one token?
        shared = sum(bool(set.intersection(*[set(x) for x in forms])) for forms in by_base.values())
        report[script] = {
            "n_chars": len(chars),
            "single_token_chars": sum(len(v) == 1 for v in ids.values()),
            "mean_tokens_per_char": sum(map(len, ids.values())) / len(ids),
            "consonants_with_shared_token_across_forms": f"{shared}/{len(by_base)}",
            "example": {ch: [tok.convert_ids_to_tokens(i) for i in [ids[ch]]][0]
                        for ch, _, _ in chars[:7]},
        }
    for name, lines in [("eth", AMHARIC_LINES), ("hangul", KOREAN_LINES), ("english", ENGLISH_LINES)]:
        n_tok = sum(len(enc(s)) for s in lines)
        n_chr = sum(len(s) for s in lines)
        report.setdefault("lines", {})[name] = {"tokens_per_char": n_tok / n_chr,
                                                "tokens_per_line": n_tok / len(lines)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "__")
    (out / f"tokenizer_{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
