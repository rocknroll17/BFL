#!/usr/bin/env bash
# Download everything needed for training and evaluation into third_party/AniMerPlus/data/.
#
#   bash scripts/download_data.sh                 # ~22 GB on disk after extraction (~33 GB peak)
#   KEEP_ARCHIVES=1 bash scripts/download_data.sh
#
# Sources
#   AniMer+ release (Google Drive folder 146ic3vnlgqutY3lh6BdV7ZXt9Ox2VAfh):
#     AniMerPlus/checkpoint.ckpt (9.4 GB), vitmoe.pth (2.55 GB), CtrlAVES3D.zip (9.1 GB)
#   CUB-200-2011 images + annotations   https://data.caltech.edu/records/65de6-vp158  (CUB_200_2011.tgz, 1.15 GB)
#   CUB-200-2011 segmentations          https://data.caltech.edu/records/w9d68-gec53  (segmentations.tgz, 39 MB)
#   CowBird (Badger et al., ECCV 2020)  https://github.com/marcbadger/avian-mesh      (cowbird.tar.gz, 913 MB)
#
# The AniMer+-format evaluation jsons (CUB/test_official.json = official CUB test split, CUB/train_split.json =
# official CUB train split used only for development, CowBird/cowbird_eval/test.json) are not in the AniMer+
# release; scripts/build_eval_jsons.py builds them from the files above.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/third_party/AniMerPlus/data"
PY="${PY:-python}"
DRIVE_FOLDER="146ic3vnlgqutY3lh6BdV7ZXt9Ox2VAfh"
COWBIRD_DRIVE_ID="1vyXYIJIo9jneIqC7lowB4GVi17rjztjn"
CUB_IMG_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_SEG_URL="https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
mkdir -p "${DATA}"
cd "${DATA}"

ok()   { echo "  [ok]   $*"; }
note() { echo "  [note] $*"; }
die()  { echo "  [FAIL] $*" >&2; exit 1; }

# ---- disk guard
free_gb=$(df -BG --output=avail "${DATA}" | tail -1 | tr -dc '0-9')
need_gb=35
[[ -f AniMerPlus/checkpoint.ckpt ]] && need_gb=$((need_gb - 10))
[[ -d control_animal3d_plus ]] && need_gb=$((need_gb - 19))
if (( free_gb < need_gb )); then
    die "only ${free_gb} GB free under ${DATA}; about ${need_gb} GB needed. Free space or make third_party/AniMerPlus/data a symlink to another volume."
fi
echo "  disk: ${free_gb} GB free, need about ${need_gb} GB"

# ---- 1. AniMer+ release (Google Drive)
if [[ -f AniMerPlus/checkpoint.ckpt && -f vitmoe.pth && -d control_animal3d_plus ]]; then
    ok "AniMer+ release already present"
else
    "${PY}" -c "import gdown" 2>/dev/null || { note "installing gdown"; "${PY}" -m pip install -q gdown; }
    mkdir -p _animerplus_release
    echo "  downloading the AniMer+ Drive folder (about 21 GB; resumable, re-run on failure)"
    "${PY}" -m gdown --folder "https://drive.google.com/drive/folders/${DRIVE_FOLDER}" \
        -O _animerplus_release --continue 2>&1 | grep -v "%|" | tail -20
    [[ ${PIPESTATUS[0]} -eq 0 ]] || die "gdown failed (Google Drive rate limit?). Retry later, or download the folder in a browser (link in third_party/AniMerPlus/README.md) into ${DATA}/_animerplus_release/"
    REL="_animerplus_release"; [[ -d "${REL}/AniMerPlus/AniMerPlus" ]] && REL="_animerplus_release/AniMerPlus"
    if [[ ! -f AniMerPlus/checkpoint.ckpt ]]; then
        mkdir -p AniMerPlus
        src_ck="$(find "${REL}" -name checkpoint.ckpt | head -1)"; [[ -n "${src_ck}" ]] || die "checkpoint.ckpt not found in the downloaded folder"
        mv "${src_ck}" AniMerPlus/checkpoint.ckpt
        ok "AniMerPlus/checkpoint.ckpt"
    fi
    if [[ ! -f vitmoe.pth ]]; then
        src_v="$(find "${REL}" -name vitmoe.pth | head -1)"; [[ -n "${src_v}" ]] || die "vitmoe.pth not found in the downloaded folder"
        mv "${src_v}" vitmoe.pth; ok "vitmoe.pth"
    fi
    if [[ ! -d control_animal3d_plus ]]; then
        src_z="$(find "${REL}" -name CtrlAVES3D.zip | head -1)"; [[ -n "${src_z}" ]] || die "CtrlAVES3D.zip not found in the downloaded folder"
        echo "  extracting CtrlAVES3D.zip -> control_animal3d_plus/"
        "${PY}" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall('.')" "${src_z}"
        [[ -f control_animal3d_plus/train.json && -f control_animal3d_plus/test.json ]] || die "CtrlAVES3D extraction incomplete"
        [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -f "${src_z}"
        ok "control_animal3d_plus/ (train.json, test.json)"
    fi
    [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -rf _animerplus_release
fi

# ---- 2. CUB-200-2011 (Caltech)
mkdir -p CUB
if [[ -d CUB/CUB_200_2011/images ]]; then ok "CUB/CUB_200_2011/images"; else
    echo "  downloading CUB_200_2011.tgz (1.15 GB)"
    curl -fL --retry 5 -C - -o CUB/CUB_200_2011.tgz "${CUB_IMG_URL}" || die "CUB_200_2011.tgz download failed"
    tar -xzf CUB/CUB_200_2011.tgz -C CUB
    [[ -d CUB/CUB_200_2011/images ]] || die "CUB_200_2011.tgz layout unexpected"
    [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -f CUB/CUB_200_2011.tgz
    ok "CUB/CUB_200_2011/images"
fi
if [[ -d CUB/segmentations ]]; then ok "CUB/segmentations"; else
    echo "  downloading segmentations.tgz (39 MB)"
    curl -fL --retry 5 -C - -o CUB/segmentations.tgz "${CUB_SEG_URL}" || die "segmentations.tgz download failed"
    tar -xzf CUB/segmentations.tgz -C CUB     # CUB/segmentations/, sibling of CUB_200_2011/
    [[ -d CUB/segmentations ]] || die "segmentations.tgz layout unexpected"
    [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -f CUB/segmentations.tgz
    ok "CUB/segmentations"
fi

# ---- 3. CowBird (Badger et al.)
if [[ -d CowBird/_badger/cowbird/annotations ]]; then ok "CowBird/_badger/cowbird"; else
    "${PY}" -c "import gdown" 2>/dev/null || "${PY}" -m pip install -q gdown
    mkdir -p CowBird/_badger
    echo "  downloading cowbird.tar.gz (913 MB)"
    "${PY}" -m gdown "${COWBIRD_DRIVE_ID}" -O CowBird/_badger/cowbird.tar.gz --continue \
        && tar -xzf CowBird/_badger/cowbird.tar.gz -C CowBird/_badger \
        && { [[ "${KEEP_ARCHIVES:-0}" == 1 ]] || rm -f CowBird/_badger/cowbird.tar.gz; ok "CowBird/_badger/cowbird"; } \
        || note "CowBird download failed (Google Drive); re-run later"
fi

# ---- 4. evaluation jsons
if [[ -f CUB/test_official.json && -f CUB/train_split.json && -f CowBird/cowbird_eval/test.json ]]; then
    ok "evaluation jsons present"
else
    echo "  building the evaluation jsons"
    "${PY}" "${ROOT}/scripts/build_eval_jsons.py" --dataset ALL
fi

echo
du -sh "${DATA}"/AniMerPlus "${DATA}"/vitmoe.pth "${DATA}"/control_animal3d_plus "${DATA}"/CUB "${DATA}"/CowBird 2>/dev/null | sed 's/^/  /'
echo "[download_data] done"
