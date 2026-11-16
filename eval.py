"""Evaluate a checkpoint on CUB, CowBird or CtrlAVES3D.

  CUB            CUB-200-2011 official test split (5,794 images)
  COWBIRD        CowBird, 114 annotated instances of unseen species
  CTRLANI3DPLUS  CtrlAVES3D test split (309 images); also reports PA-MPJPE and PA-MPVPE
  CUB_TRAIN      CUB-200-2011 official training split (development list only)

Metrics are averaged over batches of 16 images, as in AniMer+. Works for our checkpoints
(bfl.pt, or final.pt from train.py) and for the released AniMer+ checkpoint.

Usage:
    python eval.py --ckpt checkpoints/bfl.pt --dataset CUB --out-json outputs/bfl/eval_CUB.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJ = Path(__file__).resolve().parent
ANIMERPLUS = PROJ / "third_party/AniMerPlus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["CUB", "COWBIRD", "CTRLANI3DPLUS", "CUB_TRAIN"])
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    ckpt, out_json = args.ckpt.resolve(), args.out_json.resolve()

    # AniMer+ resolves its data paths relative to its own root.
    os.chdir(ANIMERPLUS)
    sys.path.insert(0, str(ANIMERPLUS))
    from amr.configs import get_config
    from amr.datasets.animal3d_dataset import EvaluationDataset
    from amr.models.animerpp import AniMerPlusPlus
    from amr.utils import recursive_to
    from amr.utils.evaluate_metric import Evaluator

    cfg = get_config(str(PROJ / "configs/animerplus.yaml"))
    eval_cfg = get_config(str(PROJ / "configs/datasets.yaml"))

    # The OU prior is training-only: its scalars (ou_prior.*) are dropped and the plain AniMer+ model is evaluated.
    # The mesh renderer is only for visualization, so evaluation needs no OpenGL.
    model = AniMerPlusPlus(cfg=cfg, init_renderer=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model_state") or ck["state_dict"]
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("ou_prior.")})
    model.cuda().eval()

    ds_cfg = eval_cfg.DATASETS[args.dataset]
    dataset = EvaluationDataset(root_image=ds_cfg.ROOT_IMAGE, json_file=ds_cfg.JSON_FILE.TEST,
                                augm_config=eval_cfg.DATASETS.CONFIG, focal_length=cfg.AVES.get("FOCAL_LENGTH", 2167),
                                image_size=cfg.MODEL.IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=2)
    evaluator = Evaluator(smal_model=model.aves, image_size=cfg.MODEL.IMAGE_SIZE)

    auc, pck, mpjpe, mpvpe = [], [], [], []
    for batch in tqdm(loader, desc=args.dataset):
        batch = recursive_to(batch, "cuda")
        with torch.no_grad():
            out = model(batch)["aves_output"]
        if args.dataset == "CTRLANI3DPLUS":
            pa_mpjpe, pa_mpvpe = evaluator.eval_3d(out, batch)
            mpjpe.append(pa_mpjpe)
            mpvpe.append(pa_mpvpe)
        p, a = evaluator.eval_2d(out, batch, pck_threshold=eval_cfg.METRIC.PCK_THRESHOLD)
        auc.append(a)
        pck.append(p)

    pck = np.array(pck)
    result = {
        "checkpoint": ckpt.parent.name + "/" + ckpt.name,
        "dataset": args.dataset,
        "n_images": len(dataset),
        "auc": float(np.mean(auc)),
        "pck_0.10": float(pck[:, 0].mean()),
        "pck_0.15": float(pck[:, 1].mean()),
    }
    if mpjpe:
        result.update(pa_mpjpe=float(np.mean(mpjpe)), pa_mpvpe=float(np.mean(mpvpe)))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
