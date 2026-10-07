# ---------------------------------------------------------------------------
# Vipsania – unsupervised deep-learning ab-initio gene finder
# https://github.com/gaius-augustus/vipsania
#
# Requires NVIDIA Container Toolkit on the host for GPU access.
# Based on NVIDIA's NGC TensorFlow image: TensorFlow 2.17.0+nv25.2 with
# CUDA 12.8 and cuDNN 9, built for all GPU generations up to Blackwell
# (sm_120). Host needs an NVIDIA driver >= 570 (data-center GPUs: older drivers
# in CUDA forward-compatibility mode, see the NGC release notes).
#
# Build:
#   sudo docker build --platform linux/amd64 -t gaiusaugustus/vipsania:1.0.0 .
#
# Run (with GPU and a local data directory):
#   sudo docker run --gpus all \
#       -v /path/to/data:/data \
#       gaiusaugustus/vipsania \
#       vipsania annotate <model_id> genome.fa -o annotation.gff3
#
# Persist downloaded models (~100 MB each) across runs:
#   sudo docker run --gpus all \
#       -v /path/to/data:/data \
#       -v /path/to/model_cache:/cache/vipsania/models \
#       gaiusaugustus/vipsania \
#       vipsania annotate <model_id> genome.fa -o annotation.gff3
# ---------------------------------------------------------------------------

FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

USER root

# Record the Python packages of the NGC base image. The check at the end of
# this file fails the build if a later pip install replaced NGC's TensorFlow or
# added PyPI CUDA/cuDNN wheels next to it (those predate Blackwell).
RUN python3 -c "import importlib.metadata as m; print('\n'.join(sorted(d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions())))" > /opt/ngc-base-packages.txt

# Vipsania uses Keras 3 (tf.keras.Layer); NGC defaults to legacy Keras 2
ENV TF_USE_LEGACY_KERAS=0
RUN python3 -m pip install --no-cache-dir --upgrade "keras>=3,<4"

LABEL org.opencontainers.image.title="Vipsania" \
      org.opencontainers.image.description="Unsupervised deep-learning ab-initio gene finder for eukaryotic genomes" \
      org.opencontainers.image.version="1.0.2" \
      org.opencontainers.image.source="https://github.com/gaius-augustus/vipsania" \
      org.opencontainers.image.authors="Richard Krieg <irkri@irkri.net>, Mario Stanke <mario.stanke@uni-greifswald.de>" \
      org.opencontainers.image.licenses="MIT"

# ── System dependencies ────────────────────────────────────────────────────
# libgomp1  : OpenMP runtime used by some TensorFlow ops
# ca-certificates, wget : for model downloads over HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install Vipsania from source ───────────────────────────────────────────
WORKDIR /opt/vipsania
COPY . /opt/vipsania/

# Not a plain `pip install .`: bricks2marble[tf] -> hidten[tensorflow]
# requires tensorflow[and-cuda], which would pull PyPI TensorFlow/CUDA wheels
# on top of the NGC stack. So Vipsania and the TF-dependent packages are
# installed without dependencies and the remaining ones explicitly.
# protobuf<5: TensorFlow 2.17 needs it; recent wandb would pull protobuf 7.
RUN python3 -m pip install --no-cache-dir --no-deps . "bricks2marble>=0.1.2" hidten \
    && python3 -m pip install --no-cache-dir numpy pydantic wandb "protobuf<5"

# ── Model cache ────────────────────────────────────────────────────────────
# Models are downloaded on first use to $VIPSANIA_CACHE/models = /cache/vipsania/models.
# Mount a host directory there to persist downloads across container runs:
#   -v /host/model_cache:/cache/vipsania/models
ENV VIPSANIA_CACHE=/cache/vipsania
RUN mkdir -p /cache/vipsania/models
VOLUME ["/cache/vipsania/models"]

# ── Working directory for user data ───────────────────────────────────────
WORKDIR /data

# Fail the build if the NGC TensorFlow/CUDA stack was modified (see top of file)
RUN python3 -c "import importlib.metadata as m, sys; \
base = set(open('/opt/ngc-base-packages.txt').read().split()); \
now = {d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions()}; \
added = sorted(p for p in now - base if p.startswith(('nvidia-', 'tensorflow'))); \
tf = m.version('tensorflow'); \
sys.exit(f'NGC TensorFlow stack modified: tensorflow=={tf}, added/changed: {added}' if '+nv' not in tf or added else 0)" \
    && python3 -c "import importlib.metadata as m, sys; from packaging.requirements import Requirement; \
reqs = [r for r in map(Requirement, m.requires('tensorflow')) if r.marker is None or r.marker.evaluate({'extra': ''})]; \
ver = lambda n: next((d.version for d in m.distributions() if d.metadata['Name'].lower().replace('_', '-') == n.lower().replace('_', '-')), None); \
bad = [f'{r} (installed: {ver(r.name)})' for r in reqs if ver(r.name) is None or not r.specifier.contains(ver(r.name), prereleases=True)]; \
sys.exit(f'TensorFlow requirements broken: {bad}' if bad else 0)" \
    && python3 -c "import tensorflow as tf, keras; b = tf.sysconfig.get_build_info(); print('TensorFlow', tf.__version__, 'CUDA', b['cuda_version'], 'cuDNN', b['cudnn_version'], 'Keras', keras.__version__)" \
    && vipsania --help > /dev/null

CMD ["vipsania", "--help"]
