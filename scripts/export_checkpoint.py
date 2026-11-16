#!/usr/bin/env python3
"""Export one of our fine-tuned checkpoints into the AniMer+ release layout.

`train.py` saves `final.pt = {"model_state": state_dict, "step", "metrics"}`.
AniMer+'s own tools (`demo.py`, `eval.py`, `amr.models.load_amr`) expect a
Lightning checkpoint with a `state_dict` key, located at
`<dir>/checkpoints/checkpoint.ckpt` next to `<dir>/.hydra/config.yaml`.
This script writes exactly that, so our checkpoints are drop-in for the
upstream demo and evaluation code:

    python scripts/export_checkpoint.py checkpoints/bfl.pt \
        --out exported/bfl
    cd third_party/AniMerPlus
    python demo.py --checkpoint ../../exported/bfl/checkpoints/checkpoint.ckpt \
        --img_folder example_data --out_folder ../../demo_out --animal_type bird

The OU prior parameters (alpha, sigma, mu) are training-time only; they are
removed from the exported state dict and saved alongside as `ou_prior.json`.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("final_pt", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output directory (gets checkpoints/ and .hydra/)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/animerplus.yaml")
    a = ap.parse_args()

    ck = torch.load(str(a.final_pt), map_location="cpu", weights_only=False)
    sd = ck.get("model_state") or ck.get("state_dict") or ck

    ou = {k: v for k, v in sd.items() if k.startswith("ou_prior.")}
    model_sd = {k: v for k, v in sd.items() if not k.startswith("ou_prior.")}

    out_ck = a.out / "checkpoints" / "checkpoint.ckpt"
    out_ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model_sd,
        "pytorch-lightning_version": "2.1.4",
        "global_step": int(ck.get("step", 0)),
        "epoch": 0,
        "source": a.final_pt.parent.name + "/" + a.final_pt.name,
        "note": "Beyond Family Labels fine-tune of AniMer+ (OU prior); exported by scripts/export_checkpoint.py",
    }, out_ck)

    (a.out / ".hydra").mkdir(exist_ok=True)
    shutil.copy(a.config, a.out / ".hydra" / "config.yaml")

    if ou:
        def sp(x):  # softplus(raw) = the positive scalar used in the loss
            return F.softplus(x).tolist()
        meta = {
            "alpha": sp(ou["ou_prior.ou.alpha_raw"]) if "ou_prior.ou.alpha_raw" in ou else None,
            "sigma": sp(ou["ou_prior.ou.sigma_raw"]) if "ou_prior.ou.sigma_raw" in ou else None,
            "mu": ou["ou_prior.ou.mu"].tolist() if "ou_prior.ou.mu" in ou else None,
            "raw_keys": sorted(ou.keys()),
        }
        json.dump(meta, open(a.out / "ou_prior.json", "w"), indent=2)

    n_params = sum(v.numel() for v in model_sd.values())
    print(f"wrote {out_ck}  ({len(model_sd)} tensors, {n_params/1e6:.1f} M values; {len(ou)} OU tensors -> ou_prior.json)")
    print(f"config -> {a.out/'.hydra'/'config.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
