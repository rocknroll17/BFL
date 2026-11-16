#!/usr/bin/env python3
"""Download the released checkpoints from the Hugging Face Hub into checkpoints/.

    python scripts/download_checkpoints.py              # all three (~3.2 GB each)
    python scripts/download_checkpoints.py bfl          # one or more by name

    checkpoints/bfl.pt                          main model (Table 1 seed 0, Table 3, Fig. 4)
    checkpoints/ablation/taxonomy_real.pt       Fig. 3 and Table 2, real D_tax
    checkpoints/ablation/taxonomy_permuted.pt   Fig. 3 and Table 2, label-permuted D_tax

Each file is checked against its md5 after download.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HF_REPO = "rocknroll17/BFL"
ROOT = Path(__file__).resolve().parents[1]

# name -> (file in the Hub repo, md5)
CHECKPOINTS = {
    "bfl":               ("bfl.pt", "0dbf86fdfd7bdfd79fe9683c474d2bba"),
    "taxonomy_real":     ("ablation/taxonomy_real.pt", "752f0694e50705a03451dc8c399bab14"),
    "taxonomy_permuted": ("ablation/taxonomy_permuted.pt", "8c3d5c01c599741d504a4fad801c9f5e"),
}


def md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help=f"any of {', '.join(CHECKPOINTS)} (default: all)")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    a = ap.parse_args()
    if unknown := set(a.names) - set(CHECKPOINTS):
        ap.error(f"unknown checkpoint(s): {', '.join(sorted(unknown))}")

    from huggingface_hub import hf_hub_download

    rc = 0
    for n in a.names or CHECKPOINTS:
        filename, ref = CHECKPOINTS[n]
        dst = a.out / filename
        if not (dst.exists() and md5(dst) == ref):
            hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=a.out)
        good = md5(dst) == ref
        rc |= not good
        print(f"[{'ok' if good else 'MD5 MISMATCH'}] {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
