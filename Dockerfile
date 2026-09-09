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
#   sudo docker build -t gaiusaugustus/vipsania:1.0.0 .
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
# Models are downloaded on first use to $VIPSANIA_CACHE/models = /cache/vipsania/models.
# Mount a host directory there to persist downloads across container runs:
#   -v /host/model_cache:/cache/vipsania/models
ENV VIPSANIA_CACHE=/cache/vipsania
RUN mkdir -p /cache/vipsania/models
VOLUME ["/cache/vipsania/models"]

# ── Entrypoint ─────────────────────────────────────────────────────────────
# Sets LD_LIBRARY_PATH to TF's bundled CUDA libs before exec-ing the
# requested command (see docs/troubleshooting.md for the rationale).
# Inlined as base64 so the Dockerfile is self-contained and does not require
# docker/entrypoint.sh to be present in the build context.
RUN echo \
    'IyEvdXNyL2Jpbi9lbnYgYmFzaAojIFZpcHNhbmlhIERvY2tlciBlbnRyeXBvaW50CiMKIyBTZXRz' \
    'IExEX0xJQlJBUllfUEFUSCB0byBUZW5zb3JGbG93J3MgYnVuZGxlZCBDVURBIGxpYnJhcmllcyBz' \
    'byB0aGUgR1BVCiMgaXMgZGlzY292ZXJlZCBjb3JyZWN0bHkgZXZlbiB3aGVuIGEgc3lzdGVtLXdp' \
    'ZGUgQ1VEQSBpbnN0YWxsYXRpb24gaXMKIyBwcmVzZW50IG9uIHRoZSBob3N0LiAgVGhpcyBtaXJy' \
    'b3JzIHRoZSBmaXggaW4gZG9jcy90cm91Ymxlc2hvb3RpbmcubWQuCiMKIyBBZnRlciBwYXRjaGlu' \
    'ZyB0aGUgcGF0aCwgdGhlIGVudHJ5cG9pbnQgZXhlYy1yZXBsYWNlcyBpdHNlbGYgd2l0aCB0aGUK' \
    'IyBjb21tYW5kIHBhc3NlZCBieSB0aGUgY2FsbGVyIChlLmcuIHZpcHNhbmlhIGFubm90YXRlIC4u' \
    'LikuCgpzZXQgLWV1byBwaXBlZmFpbAoKIyBDb2xsZWN0IGFsbCBudmlkaWEvKi9saWIgcGF0aHMg' \
    'aW5zdGFsbGVkIGJ5IHRoZSBURiBwaXAgcGFja2FnZXMuClRGX0NVREE9JChweXRob24gLWMgImlt' \
    'cG9ydCBzaXRlLCBnbG9iLCBvczsgcHJpbnQoJzonLmpvaW4oc29ydGVkKGdsb2IuZ2xvYihvcy5w' \
    'YXRoLmpvaW4oc2l0ZS5nZXRzaXRlcGFja2FnZXMoKVswXSwgJ252aWRpYScsICcqJywgJ2xpYicp' \
    'KSkpKSIgMj4vZGV2L251bGwgfHwgdHJ1ZSkKCmlmIFsgLW4gIiR7VEZfQ1VEQX0iIF07IHRoZW4K' \
    'ICAgIGV4cG9ydCBMRF9MSUJSQVJZX1BBVEg9IiR7VEZfQ1VEQX0ke0xEX0xJQlJBUllfUEFUSDor' \
    'OiR7TERfTElCUkFSWV9QQVRIfX0iCmZpCgpleGVjICIkQCIK' \
    | tr -d ' \n' | base64 -d > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# ── Working directory for user data ───────────────────────────────────────
WORKDIR /data

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["vipsania", "--help"]
