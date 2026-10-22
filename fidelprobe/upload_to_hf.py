"""Creates the public HF bucket and uploads README, code, data, results and the model."""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def build_readme(dst):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tables = sorted((ROOT / "results").glob("RESULTS_*.md"))
    if tables and "<!-- RESULTS -->" in readme:
        readme = readme.replace("<!-- RESULTS -->",
                                "\n\n".join(t.read_text(encoding="utf-8") for t in tables))
    (dst / "README.md").write_text(readme, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="fidelprobe", help="bucket name, or namespace/name")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--no-model", action="store_true", help="do not copy the model weights")
    args = ap.parse_args()

    api = HfApi()
    bucket_id = args.bucket if "/" in args.bucket else f"{api.whoami()['name']}/{args.bucket}"
    url = api.create_bucket(bucket_id, visibility="public", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        build_readme(stage)
        for sub in ["fidelprobe", "data", "results"]:
            shutil.copytree(ROOT / sub, stage / sub,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if not args.no_model:
            snapshot_download(args.model, local_dir=stage / "model",
                              ignore_patterns=[".cache/*"])
            shutil.rmtree(stage / "model" / ".cache", ignore_errors=True)
        api.sync_bucket(str(stage), f"hf://buckets/{bucket_id}")
    print(f"Uploaded to https://huggingface.co/buckets/{bucket_id}  ({url})")


if __name__ == "__main__":
    main()
