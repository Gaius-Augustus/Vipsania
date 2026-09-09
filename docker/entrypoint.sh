#!/usr/bin/env bash
# Vipsania Docker entrypoint
#
# Sets LD_LIBRARY_PATH to TensorFlow's bundled CUDA libraries so the GPU
# is discovered correctly even when a system-wide CUDA installation is
# present on the host.  This mirrors the fix in docs/troubleshooting.md.
#
# After patching the path, the entrypoint exec-replaces itself with the
# command passed by the caller (e.g. vipsania annotate ...).

set -euo pipefail

# Collect all nvidia/*/lib paths installed by the TF pip packages.
TF_CUDA=$(python -c "import site, glob, os; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))" 2>/dev/null || true)

if [ -n "${TF_CUDA}" ]; then
    export LD_LIBRARY_PATH="${TF_CUDA}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "$@"
