"""Fine-tune AniMer+ with the taxonomic OU prior (one variant of configs/variants.yaml).

The training logic is the one that produced every checkpoint in the paper;
only the unused switches of the original hyper-parameter sweep were removed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train.py --variant table1_seed0 --out-dir outputs/table1_seed0

Writes <out-dir>/final.pt ({"step", "model_state", "metrics"}), metrics.json and,
unless SKIP_INTERMEDIATE_CKPT=1, a checkpoint every TOTAL_STEPS/4 steps.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJ = Path(__file__).resolve().parent
ANIMERPLUS = PROJ / "third_party/AniMerPlus"


def main():
    sys.stdout.reconfigure(line_buffering=True)   # progress lines appear immediately when logging to a file
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()    # the script chdirs into repos/AniMerPlus below
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = yaml.safe_load(open(PROJ / "configs/variants.yaml"))
    common = registry["common"]
    v = registry["variants"][args.variant]
    print(f"[{args.variant}] config: {v}")

    seed = int(v.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[{args.variant}] seed={seed}")

    # AniMer+ resolves its data paths relative to its own root.
    os.chdir(ANIMERPLUS)
    sys.path.insert(0, str(ANIMERPLUS))

    from amr.configs import get_config
    from amr.datasets import AniMerPlusPlusDataModule
    from amr.utils import recursive_to
    from bfl.datasets import use_species_labels
    from bfl.model import BFLModel

    use_species_labels()          # before the data module is set up (see bfl/datasets.py)

    cfg = get_config(str(PROJ / "configs/animerplus.yaml"))
    cfg.defrost()
    cfg.GENERAL.TOTAL_STEPS = int(v.get("total_steps", common["total_steps"]))
    cfg.GENERAL.LOG_STEPS = int(common["log_steps"])
    cfg.TRAIN.LR = float(common["lr"])
    cfg.TRAIN.BATCH_SIZE = int(v.get("batch_size", common["batch_size"]))
    cfg.GENERAL.NUM_WORKERS = int(common.get("num_workers", 4))
    cfg.LOSS_WEIGHTS.SUPCON = float(v["supcon_weight"])     # AniMer+'s family-contrastive loss; 0 = off
    print(f"[{args.variant}] total_steps={cfg.GENERAL.TOTAL_STEPS} batch_size={cfg.TRAIN.BATCH_SIZE} lr={cfg.TRAIN.LR}")
    cfg.freeze()

    ou_weight = float(v["ou_weight"])
    phylo_path = v.get("phylo_matrix_override") or common["phylo_matrix"]
    # Warm start from the released AniMer+ checkpoint; the OU prior's scalars are new (strict=False).
    model = BFLModel.load_from_checkpoint(
        str(PROJ / common["pretrained_checkpoint"]), cfg=cfg, strict=False, ou_weight=ou_weight,
        phylo_matrix_path=str(PROJ / phylo_path), cub_mapping_path=str(PROJ / common["cub_mapping"]))
    model.cuda()
    model.train()

    dm = AniMerPlusPlusDataModule(cfg)
    dm.setup("fit")
    dl_dict = dm.train_dataloader()
    dl = dl_dict["img"] if isinstance(dl_dict, dict) else dl_dict

    # Two parameter groups: the network, and the OU scalars (alpha, sigma, mu) at k x LR.
    ou_params = list(model.ou_prior.ou.parameters()) if ou_weight > 0 else []
    ou_ids = {id(p) for p in ou_params}
    main_params = [p for p in model.parameters() if p.requires_grad and id(p) not in ou_ids]
    ou_lr_mult = float(v.get("ou_lr_multiplier", 0.1))
    ou_weight_decay = float(v.get("ou_weight_decay", 1e-4))
    print(f"[{args.variant}] ou_lr_multiplier={ou_lr_mult} (ou_lr={cfg.TRAIN.LR * ou_lr_mult:.2e}), ou_wd={ou_weight_decay}")
    optim = torch.optim.AdamW(
        [
            {"params": main_params, "lr": cfg.TRAIN.LR, "weight_decay": 1e-4},
            {"params": ou_params, "lr": cfg.TRAIN.LR * ou_lr_mult, "weight_decay": ou_weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = None
    if bool(v.get("use_cosine_lr", False)):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optim, T_max=cfg.GENERAL.TOTAL_STEPS, eta_min=cfg.TRAIN.LR * 0.01)
        print(f"[{args.variant}] cosine LR: eta_min={cfg.TRAIN.LR * 0.01:.2e}")

    metrics = {"variant": args.variant, "steps": [], "loss": [], "alpha": [], "sigma": []}
    step = 0
    loss_ema = None
    skip_intermediate = os.environ.get("SKIP_INTERMEDIATE_CKPT", "0") == "1"
    for _epoch in range(100):  # bounded by TOTAL_STEPS
        for batch in dl:
            if step >= cfg.GENERAL.TOTAL_STEPS:
                break
            batch = recursive_to(batch, "cuda")
            out = model(batch)
            loss = model.compute_loss(batch, out)
            if not torch.isfinite(loss):
                print(f"[{args.variant}] step {step}: loss non-finite. Abort.")
                sys.exit(3)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params + ou_params, max_norm=1.0)
            optim.step()
            if scheduler is not None:
                scheduler.step()

            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()

            if step % cfg.GENERAL.LOG_STEPS == 0:
                a = model.ou_prior.alpha().item()
                s = model.ou_prior.sigma().item()
                metrics["steps"].append(step)
                metrics["loss"].append(loss_ema)
                metrics["alpha"].append(a)
                metrics["sigma"].append(s)
                print(f"[{args.variant}][step {step:5d}] loss_ema={loss_ema:.4f}  alpha={a:.4f}  sigma={s:.4f}")
                json.dump(metrics, open(args.out_dir / "metrics.json", "w"), indent=2)

            if (not skip_intermediate) and step > 0 and step % (cfg.GENERAL.TOTAL_STEPS // 4) == 0:
                ckpt_path = args.out_dir / f"ckpt_step{step}.pt"
                torch.save({"step": step, "model_state": model.state_dict(),
                            "optim_state": optim.state_dict(), "metrics": metrics}, ckpt_path)
                print(f"[{args.variant}] checkpoint saved: {ckpt_path}")

            step += 1
        if step >= cfg.GENERAL.TOTAL_STEPS:
            break

    final_path = args.out_dir / "final.pt"
    torch.save({"step": step, "model_state": model.state_dict(), "metrics": metrics}, final_path)
    json.dump(metrics, open(args.out_dir / "metrics.json", "w"), indent=2)
    print(f"[{args.variant}] done. step={step}  saved {final_path}")


if __name__ == "__main__":
    main()
