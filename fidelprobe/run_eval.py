"""Run FidelProbe against a Hugging Face vision-language model.

Greedy decoding, one item at a time, resumable (already-answered ids are
skipped). Works on CPU, a Colab T4, or Modal.

  python run_eval.py --model Qwen/Qwen3-VL-2B-Instruct --data ../data --out ../results
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM = ("You are taking a test. Reply with the final answer only, exactly in the requested "
          "format. Do not explain, do not repeat the question.")
MAX_NEW = {"char": 12, "digit": 6, "yesno": 4, "text": 96}


def load(model_id):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    proc = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, proc


@torch.inference_mode()
def generate(model, proc, item, data_dir):
    content = []
    if item.get("image"):
        content.append({"type": "image", "image": Image.open(data_dir / item["image"]).convert("RGB")})
    content.append({"type": "text", "text": item["prompt"]})
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer:"}]}]
    # Prefill "Answer:" so the model commits to an answer instead of restating the question.
    inputs = proc.apply_chat_template(msgs, tokenize=True, continue_final_message=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW[item["answer_type"]], do_sample=False)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--tasks", default="", help="comma-separated subset of tasks")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data)
    items = [json.loads(l) for l in open(data_dir / "probes.jsonl", encoding="utf-8")]
    if args.tasks:
        keep = set(args.tasks.split(","))
        items = [it for it in items if it["task"] in keep]
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_{args.model.replace('/', '__')}.jsonl"
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["id"] for l in open(pred_path, encoding="utf-8")}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(todo)} to run", flush=True)

    torch.manual_seed(0)
    model, proc = load(args.model)
    t0 = time.time()
    with open(pred_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo, 1):
            pred = generate(model, proc, it, data_dir)
            f.write(json.dumps({"id": it["id"], "model": args.model, "pred": pred},
                               ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                rate = (time.time() - t0) / n
                print(f"{n}/{len(todo)}  {rate:.1f}s/item  eta {rate * (len(todo) - n) / 60:.0f} min",
                      flush=True)


if __name__ == "__main__":
    main()
