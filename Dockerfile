# ---------------------------------------------------------------------------
# Vipsania – unsupervised deep-learning ab-initio gene finder
# https://github.com/gaius-augustus/vipsania
#
# Requires NVIDIA Container Toolkit on the host for GPU access.
# TensorFlow ships its own CUDA libraries; LD_LIBRARY_PATH is set by the
# entrypoint so TF reliably finds them even on hosts that already have a
# separate CUDA installation (see docs/troubleshooting.md).
#
# Build:
#   sudo docker build -t katharinahoff/vipsania:1.0.0 .
#
# Run (with GPU and a local data directory):
#   sudo docker run --gpus all \
#       -v /path/to/data:/data \
#       katharinahoff/vipsania \
#       vipsania annotate <model_id> genome.fa -o annotation.gff3
#
# Persist downloaded models (~100 MB each) across runs:
#   sudo docker run --gpus all \
#       -v /path/to/data:/data \
#       -v /path/to/model_cache:/cache/vipsania/models \
#       katharinahoff/vipsania \
#       vipsania annotate <model_id> genome.fa -o annotation.gff3
# ---------------------------------------------------------------------------

FROM python:3.12-slim

LABEL org.opencontainers.image.title="Vipsania" \
      org.opencontainers.image.description="Unsupervised deep-learning ab-initio gene finder for eukaryotic genomes" \
      org.opencontainers.image.version="1.0.0" \
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

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ── Model cache ────────────────────────────────────────────────────────────
# Models are downloaded on first use to /cache/vipsania/models.
# Mount a host directory there to persist downloads across container runs:
#   -v /host/model_cache:/cache/vipsania/models
ENV VIPSANIA_CACHE=/cache/vipsania/models
RUN mkdir -p /cache/vipsania/models
VOLUME ["/cache/vipsania/models"]

# ── Entrypoint ─────────────────────────────────────────────────────────────
# Sets LD_LIBRARY_PATH to TF's bundled CUDA libs before exec-ing the
# requested command (see docs/troubleshooting.md for the rationale).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ── Working directory for user data ───────────────────────────────────────
WORKDIR /data

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["vipsania", "--help"]
