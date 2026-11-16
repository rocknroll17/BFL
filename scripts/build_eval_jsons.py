#!/usr/bin/env python3
"""Build the AniMer+-format evaluation jsons for CUB and CowBird from the public sources.

AniMer+ evaluates CUB and CowBird through two json files that are not part of
its public release. Their content is public:

  CUB      CUB-200-2011 in image-id order, with the official 15 part locations
           (minus 1, i.e. 0-based; invisible parts -> -1, -1) and bounding boxes
           [x, y, w, h]; masks = the official segmentations/. Two lists:
             CUB        official TEST split (train_test_split.txt == 0; 5 794 images).
                        All CUB numbers in the paper are on this list.
             CUB_TRAIN  official TRAIN split (== 1; 5 994 images). Used only as the
                        development list for choosing lambda_OU (paper, appendix).
  CowBird  the 114 keypoint-annotated instances of Badger et al. (ECCV 2020),
           cowbird/annotations/instance_test.json, in annotation order; masks
           rasterised from the COCO polygons.

Keypoints are placed in the 18-slot AVES order; CUB parts follow AniMer+'s
cub17_dataset mapping, CowBird: bill tip -> 9, eyes -> 0/1, nape -> 4,
wing tips -> 16/17, feet -> 10/12, tail tip -> 15 (neck and wrists unused).

Output (relative to third_party/AniMerPlus/data/):
  CUB/test_official.json, CUB/train_split.json
  CowBird/cowbird -> _badger/cowbird (images); CowBird/cowbird/masks/mask_<ann_id>.png
  CowBird/cowbird_eval/test.json

Usage:
  python scripts/build_eval_jsons.py --dataset ALL
  python scripts/build_eval_jsons.py --dataset ALL --verify   # GPU: evaluate the released AniMer+
                                                              # checkpoint on the result and compare with REFERENCE
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party/AniMerPlus/data"

# AniMer+ amr/datasets/cub17_dataset.py: AVES slot -> CUB part index (0-based), -1 = empty
CUB_TO_AVES = [6, 10, 5, 4, 9, 0, 14, 3, 1, -1, -1, 7, -1, 11, 13, -1, -1, -1]
# Badger CowBird keypoint names (instance_*.json categories[0].keypoints) -> AVES slot
BADGER12 = ["bill tip", "left eye", "right eye", "neck", "nape", "left wing tip", "right wing tip",
            "left foot", "right foot", "tail tip", "left wrist", "right wrist"]
COWBIRD_TO_AVES = {"bill tip": 9, "left eye": 0, "right eye": 1, "nape": 4, "left wing tip": 16,
                   "right wing tip": 17, "left foot": 10, "right foot": 12, "tail tip": 15}
SUPERCATEGORY_BIRD = 6
# Scores of the released AniMer+ checkpoint on these lists (eval.py aggregation:
# mean of per-batch means at batch size 16); --verify compares a rebuilt json against them.
REFERENCE = {
    "CUB":       {"auc": 0.896215, "pck_0.10": 0.597507},
    "CUB_TRAIN": {"auc": 0.900499, "pck_0.10": 0.623030},
    "COWBIRD":   {"auc": 0.593683, "pck_0.10": 0.164053},
}
SPLIT_NAME = {0: "CUB-200-2011 official test split (train_test_split.txt == 0)",
              1: "CUB-200-2011 official training split (train_test_split.txt == 1)"}


def record(img_path, mask_path, bbox_xywh, kp18, category, pose_len, shape_len, shape_extra_len=None):
    """One evaluation record in AniMer+'s format. The 3D fields are placeholders
    (zeros, bone = ones) and do not influence the 2D metrics."""
    r = {
        "img_path": img_path,
        "mask_path": mask_path,
        "bbox": [float(v) for v in bbox_xywh],
        "keypoint_2d": [[float(x), float(y), float(c)] for x, y, c in kp18],
        "keypoint_3d": [[0.0, 0.0, 0.0] for _ in range(18)],
        "pose": [0.0] * pose_len,
        "shape": [0.0] * shape_len,
        "trans": [0.0, 0.0, 0.0],
        "bone": [1.0] * 24,
        "supercategory": SUPERCATEGORY_BIRD,
        "category": int(category),
    }
    if shape_extra_len:
        r["shape_extra"] = [0.0] * shape_extra_len
    return r


# ---------------------------------------------------------------------- CUB
def read_cub_tables(cub_root: Path):
    def rows(name):
        with open(cub_root / name) as f:
            return [ln.split() for ln in f if ln.strip()]
    images = {int(i): p for i, p in rows("images.txt")}
    split = {int(i): int(s) for i, s in rows("train_test_split.txt")}
    bboxes = {int(r[0]): [float(v) for v in r[1:5]] for r in rows("bounding_boxes.txt")}
    parts = {}
    for r in rows("parts/part_locs.txt"):
        img, part, x, y, vis = int(r[0]), int(r[1]), float(r[2]), float(r[3]), int(r[4])
        parts.setdefault(img, [None] * 15)[part - 1] = (x, y, vis)
    return images, split, bboxes, parts


def build_cub(out: Path, split_value: int) -> int:
    """split_value: train_test_split.txt flag to keep (0 = official test, 1 = official train)."""
    cub_root = DATA / "CUB/CUB_200_2011"
    seg_root = DATA / "CUB/segmentations"
    assert (cub_root / "images.txt").exists(), f"missing {cub_root}/images.txt (extract CUB_200_2011.tgz)"
    assert seg_root.exists(), f"missing {seg_root} (extract segmentations.tgz into data/CUB/)"
    images, split, bboxes, parts = read_cub_tables(cub_root)
    recs = []
    for img_id in sorted(images):
        if split[img_id] != split_value:
            continue
        rel = images[img_id]
        kp = []
        for slot in range(18):
            ci = CUB_TO_AVES[slot]
            if ci < 0:
                kp.append((0.0, 0.0, 0.0))
                continue
            x, y, vis = parts[img_id][ci]
            kp.append((x - 1, y - 1, float(vis)))
        recs.append(record(f"images/{rel}", f"../segmentations/{Path(rel).with_suffix('.png').as_posix()}",
                           bboxes[img_id], kp, category=0, pose_len=75, shape_len=15))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"data": recs, "source": SPLIT_NAME[split_value]}, open(out, "w"))
    return len(recs)


# ------------------------------------------------------------------ CowBird
def polygon_mask(seg, w, h):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for poly in seg:
        d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=255)
    return np.array(im, dtype=np.uint8)


def build_cowbird(out: Path) -> int:
    badger = DATA / "CowBird/_badger/cowbird"
    ann_file = badger / "annotations/instance_test.json"
    assert ann_file.exists(), f"missing {ann_file} (scripts/download_data.sh fetches cowbird.tar.gz)"
    link = DATA / "CowBird/cowbird"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(os.path.relpath(badger, link.parent), link)
        except OSError:  # no symlink privilege
            shutil.copytree(badger, link)
    d = json.load(open(ann_file))
    assert d["categories"][0]["keypoints"] == BADGER12, d["categories"][0]["keypoints"]
    imgs = {im["id"]: im for im in d["images"]}
    mask_dir = link / "masks"            # mask_path is resolved relative to ROOT_IMAGE = CowBird/cowbird
    mask_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for a in d["annotations"]:
        if not a.get("keypoints") or sum(a["keypoints"][2::3]) == 0:
            continue
        im = imgs[a["image_id"]]
        kp = [(0.0, 0.0, 0.0)] * 18
        for j, name in enumerate(BADGER12):
            slot = COWBIRD_TO_AVES.get(name)
            if slot is None:
                continue
            x, y, v = a["keypoints"][3 * j:3 * j + 3]
            if v > 0:
                kp[slot] = (float(x), float(y), 1.0)
        mask_name = f"mask_{a['id']}.png"
        if not (mask_dir / mask_name).exists():
            Image.fromarray(polygon_mask(a["segmentation"], im["width"], im["height"])).save(mask_dir / mask_name)
        recs.append(record(f"images/{im['file_name']}", f"masks/{mask_name}",
                           a["bbox"], kp, category=6, pose_len=105, shape_len=26, shape_extra_len=15))
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"data": recs, "source": "Badger et al. 2020 instance_test.json"}, open(out, "w"))
    return len(recs)


# ------------------------------------------------------------------- verify
def verify(ds: str) -> bool:
    """Evaluate the released AniMer+ checkpoint on the rebuilt json and compare
    with its reference scores."""
    ref = REFERENCE[ds]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "e.json"
        subprocess.run([sys.executable, str(ROOT / "eval.py"), "--ckpt", str(DATA / "AniMerPlus/checkpoint.ckpt"),
                        "--dataset", ds, "--out-json", str(out)], check=True, cwd=ROOT)
        got = json.load(open(out))
    err = abs(got["auc"] - ref["auc"]) + abs(got["pck_0.10"] - ref["pck_0.10"])
    ok = err < 1e-3   # evaluation drift between GPU models is below 0.01 pp per metric
    print(f"[verify {ds}] AUC {100*got['auc']:.3f} (ref {100*ref['auc']:.3f})  PCK@0.1 {100*got['pck_0.10']:.3f} "
          f"(ref {100*ref['pck_0.10']:.3f})  -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["CUB", "CUB_TRAIN", "COWBIRD", "ALL"], default="ALL")
    ap.add_argument("--verify", action="store_true", help="evaluate the released AniMer+ checkpoint on the result (GPU)")
    ap.add_argument("--force", action="store_true", help="rebuild even if the json exists")
    a = ap.parse_args()

    targets = {"CUB": (DATA / "CUB/test_official.json", lambda out: build_cub(out, split_value=0)),
               "CUB_TRAIN": (DATA / "CUB/train_split.json", lambda out: build_cub(out, split_value=1)),
               "COWBIRD": (DATA / "CowBird/cowbird_eval/test.json", build_cowbird)}
    rc = 0
    for ds in (list(targets) if a.dataset == "ALL" else [a.dataset]):
        target, builder = targets[ds]
        if target.exists() and not a.force:
            print(f"[{ds}] {target} exists; keeping it (--force to rebuild)")
        else:
            n = builder(target)
            print(f"[{ds}] wrote {n} records -> {target}")
        if a.verify and not verify(ds):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
