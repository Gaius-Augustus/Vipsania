#!/usr/bin/env bash
# Build the Vipsania Docker image and push it to a Docker registry.
#
# Usage (run from anywhere; the script changes to the repo root itself):
#   bash docker/build_and_push.sh <dockerhub_user> [image_name] [version]
#
# Arguments:
#   dockerhub_user  Docker Hub username to log in as and to prefix the image
#                   (required).  Example: katharinahoff
#   image_name      Repository / image name (default: vipsania).
#                   Example: vipsania
#   version         Version tag (default: value from pyproject.toml, or
#                   "latest" if that cannot be parsed).
#                   Example: 1.0.0
#
# Examples:
#   bash docker/build_and_push.sh katharinahoff
#   bash docker/build_and_push.sh katharinahoff vipsania
#   bash docker/build_and_push.sh katharinahoff vipsania 1.0.0
#
# Prerequisites:
#   1. Docker is installed.
#   2. sudo privileges are available.
#   3. You are logged in to Docker Hub:
#        sudo docker login --username <dockerhub_user>
#      (Docker Hub token or password will be prompted once.)

set -euo pipefail

# ── Arguments ──────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <dockerhub_user> [image_name] [version]" >&2
    echo "  Example: $0 katharinahoff vipsania 1.0.0" >&2
    exit 1
fi

DOCKERHUB_USER="${1}"
IMAGE_NAME="${2:-vipsania}"
IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}"

# Resolve repo root first so we can read pyproject.toml for the version.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse version from pyproject.toml if not supplied as argument.
if [[ $# -ge 3 ]]; then
    VERSION="${3}"
else
    VERSION=$(grep -E '^version\s*=' "${REPO_ROOT}/pyproject.toml" \
               | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')
    VERSION="${VERSION:-latest}"
fi

echo "==> Repository root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

# ── Verify Docker is available ─────────────────────────────────────────────
if ! sudo docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running or not reachable via sudo." >&2
    exit 1
fi

# ── Build ──────────────────────────────────────────────────────────────────
echo "==> Building ${IMAGE}:${VERSION} and ${IMAGE}:latest ..."
sudo docker build \
    --tag "${IMAGE}:${VERSION}" \
    --tag "${IMAGE}:latest" \
    --file Dockerfile \
    .

echo "==> Build complete."

# ── Push ───────────────────────────────────────────────────────────────────
echo "==> Pushing ${IMAGE}:${VERSION} ..."
sudo docker push "${IMAGE}:${VERSION}"

echo "==> Pushing ${IMAGE}:latest ..."
sudo docker push "${IMAGE}:latest"

echo ""
echo "==> Done. Images available at:"
echo "    https://hub.docker.com/r/${IMAGE}"
echo "    Tags pushed: ${VERSION}, latest"
echo ""
echo "    To pull: sudo docker pull ${IMAGE}:${VERSION}"
