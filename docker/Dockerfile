# Environment for training and evaluation (Python 3.10, PyTorch 2.4.0 + CUDA 12.1, PyTorch3D 0.7.8).
# Build from the repository root after `git clone --recursive`:
#   docker build -f docker/Dockerfile -t bfl .
# Data and outputs are mounted at run time (see docker/docker-compose.yml and README.md).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        git curl ca-certificates build-essential ninja-build \
        libgl1 libegl1 libgles2 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0

# PyTorch, then PyTorch3D built against it (covers V100 ... H100 and RTX 30/40 GPUs).
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
ENV FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
RUN pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

WORKDIR /workspace/BFL
COPY requirements.txt .
# chumpy from git: the PyPI release (0.70) does not import with NumPy >= 1.24.
RUN pip install -r requirements.txt \
    && pip install --no-build-isolation "git+https://github.com/mattloper/chumpy.git@51d5afd92a8ded3637553be8cef41f328a1c863a"

# Code; the AniMer+ submodule must be checked out (git clone --recursive, or git submodule update --init).
COPY . .
RUN test -f third_party/AniMerPlus/amr/models/animerpp.py \
    || (echo "third_party/AniMerPlus is empty: run 'git submodule update --init' before building" && exit 1)

CMD ["bash"]
