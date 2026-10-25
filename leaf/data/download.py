"""Download ImageNet-1k as WebDataset shards (timm/imagenet-1k-wds, about 150 GB) from the Hugging Face Hub.
Restored from legacy/scripts/download_data.py (deleted in a8c577c).

Accept the ImageNet licence on https://huggingface.co/datasets/timm/imagenet-1k-wds first, then put a READ token
in .env (HF_TOKEN=...) or in the environment.

    python -m leaf data download [--out data/imagenet_tar]
"""
import argparse
import os


def download(data_dir="./data/imagenet_tar", max_workers=10):
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    # Load the variables of a .env file into the environment, if there is one.
    load_dotenv()

    print("Downloading ImageNet-1k (about 150 GB)...")

    # Token loaded from .env or the environment
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        raise ValueError("No valid token: copy .env.example to .env and fill in your Hugging Face token.")

    os.makedirs(data_dir, exist_ok=True)

    snapshot_download(
        repo_id="timm/imagenet-1k-wds",
        repo_type="dataset",
        local_dir=data_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=max_workers,
        token=HF_TOKEN
    )

    print(f"Download finished: {data_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="./data/imagenet_tar")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    download(args.out, args.workers)


if __name__ == "__main__":
    main()
