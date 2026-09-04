# Running Vipsania in a container

Pre-built images for Vipsania are published on Docker Hub:

```
docker.io/katharinahoff/vipsania          # latest release
docker.io/katharinahoff/vipsania:1.0.0   # pinned version
```

They bundle Vipsania together with all Python dependencies, including TensorFlow
with its own CUDA libraries, so **no CUDA installation on the host is required**.
GPU access is provided by the container runtime, not by a host CUDA stack.

---

## Docker

### Requirements

- Docker Engine ≥ 20.10
- GPU runs additionally require the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed on the host
- Most Docker installations need `sudo` unless your user is in the `docker`
  group; the commands below include `sudo` throughout

### Pull the image

```bash
sudo docker pull katharinahoff/vipsania:latest
```

### Annotate a genome — with GPU (recommended)

Mount the directory that contains your genome into the container at `/data`.
The container's working directory is `/data`, so relative paths in the
command refer to it.

```bash
sudo docker run --rm --gpus all \
    -v /path/to/your/data:/data \
    katharinahoff/vipsania \
    vipsania annotate Fungi genome.fa -o annotation.gff3 --finetune
```

`--gpus all` exposes every GPU that is visible to the Docker daemon.  On a
workstation this is usually what you want.  **Inside a SLURM job**, use
`--gpus "device=${CUDA_VISIBLE_DEVICES}"` instead so the container sees only
the GPU(s) SLURM allocated to your job, regardless of how many GPUs the node
has in total:

```bash
# Inside a SLURM job that requested --gres=gpu:1
sudo docker run --rm --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    -v /path/to/your/data:/data \
    katharinahoff/vipsania \
    vipsania annotate Fungi genome.fa -o annotation.gff3 --finetune
```

### Annotate a genome — CPU only

Drop `--gpus ...` entirely. Annotation on CPU is possible but slow for large
genomes.

```bash
sudo docker run --rm \
    -v /path/to/your/data:/data \
    katharinahoff/vipsania \
    vipsania annotate Fungi genome.fa -o annotation.gff3
```

### Persist the model cache

Pre-trained models (~100 MB each) are downloaded on first use.  By default
they land inside the container and are lost when it exits.  Bind-mount a host
directory to keep them across runs:

```bash
sudo docker run --rm --gpus all \
    -v /path/to/your/data:/data \
    -v /path/to/model_cache:/cache/vipsania/models \
    katharinahoff/vipsania \
    vipsania annotate Fungi genome.fa -o annotation.gff3 --finetune
```

You can also change the cache path with the environment variable
`VIPSANIA_CACHE`. Vipsania appends `/models` to it, so the setting below stores
the models in `/data/models` on the mounted volume:

```bash
sudo docker run --rm --gpus all \
    -v /path/to/your/data:/data \
    -e VIPSANIA_CACHE=/data \
    katharinahoff/vipsania \
    vipsania annotate Fungi genome.fa -o annotation.gff3 --finetune
```

### Training

Pass a config file and the genome FASTAs through the mounted volume.  Edit
`configs/train.json` to point `dataset.train_paths` at the FASTA files under
`/data`, then:

```bash
sudo docker run --rm --gpus all \
    -v /path/to/your/data:/data \
    -v /path/to/model_cache:/cache/vipsania/models \
    katharinahoff/vipsania \
    vipsania train /data/configs/base_10M.json -oc /data/configs/train.json
```

Checkpoints are written inside the container.  Map a host directory to
`/data/checkpoints` (or whichever path your config uses) if you want to keep
them:

```bash
sudo docker run --rm --gpus all \
    -v /path/to/your/data:/data \
    -v /path/to/checkpoints:/data/checkpoints \
    -v /path/to/model_cache:/cache/vipsania/models \
    katharinahoff/vipsania \
    vipsania train /data/configs/base_10M.json -oc /data/configs/train.json
```

### Interactive shell

```bash
sudo docker run --rm -it --gpus all \
    -v /path/to/your/data:/data \
    katharinahoff/vipsania \
    bash
```

---

## Singularity / Apptainer

[Singularity](https://sylabs.io/singularity/) (and its successor
[Apptainer](https://apptainer.org/)) run containers without root privileges,
which makes them the standard choice on HPC clusters.

### Convert the Docker image to a Singularity image file (SIF)

```bash
singularity pull vipsania.sif docker://katharinahoff/vipsania:latest
```

This downloads and converts the image once; the resulting `vipsania.sif` file
is portable and can be copied to any machine or cluster that has Singularity.

Pin a specific release to keep your results reproducible:

```bash
singularity pull vipsania_1.0.0.sif docker://katharinahoff/vipsania:1.0.0
```

### Annotate a genome — with GPU (recommended)

`--nv` forwards the host NVIDIA drivers into the container.  The host does
**not** need a matching CUDA installation — only the driver itself is needed.

```bash
singularity exec --nv \
    -B /path/to/your/data:/data \
    vipsania.sif \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3 --finetune
```

Or, if you want the container's working directory to be your data directory:

```bash
singularity exec --nv \
    --pwd /data \
    -B /path/to/your/data:/data \
    vipsania.sif \
    vipsania annotate Fungi genome.fa -o annotation.gff3 --finetune
```

### Annotate a genome — CPU only

Drop `--nv`:

```bash
singularity exec \
    -B /path/to/your/data:/data \
    vipsania.sif \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3
```

### Model cache

Singularity containers are read-only, so Vipsania cannot write to the default
cache path (`/cache/vipsania/models`) inside the image.  Choose a writable
location with `SINGULARITYENV_VIPSANIA_CACHE` (or `APPTAINERENV_VIPSANIA_CACHE`
on Apptainer). Models are stored in `$VIPSANIA_CACHE/models`:

```bash
# models end up in /path/to/model_cache/models
export SINGULARITYENV_VIPSANIA_CACHE=/path/to/model_cache
# or for Apptainer:
export APPTAINERENV_VIPSANIA_CACHE=/path/to/model_cache

singularity exec --nv \
    -B /path/to/your/data:/data \
    -B /path/to/model_cache:/path/to/model_cache \
    vipsania.sif \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3 --finetune
```

Alternatively, bind-mount a host directory to `/cache`:

```bash
singularity exec --nv \
    -B /path/to/your/data:/data \
    -B /path/to/model_cache:/cache/vipsania/models \
    vipsania.sif \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3 --finetune
```

### Training with Singularity

```bash
singularity exec --nv \
    -B /path/to/your/data:/data \
    -B /path/to/checkpoints:/data/checkpoints \
    -B /path/to/model_cache:/cache/vipsania/models \
    vipsania.sif \
    vipsania train /data/configs/base_10M.json -oc /data/configs/train.json
```

### HPC job script example (SLURM)

Singularity is the natural choice on HPC clusters because it does not require
root.  `--nv` reads `CUDA_VISIBLE_DEVICES` from the environment, so SLURM's
GPU allocation is automatically respected — if you request one GPU and the
node has four, the container sees exactly the one GPU that was assigned to
your job.

```bash
#!/bin/bash
#SBATCH --job-name=vipsania
#SBATCH --gres=gpu:1          # request exactly one GPU
#SBATCH --mem=32G
#SBATCH --time=4:00:00

singularity exec --nv \
    -B "$SCRATCH/data:/data" \
    -B "$SCRATCH/models:/cache/vipsania/models" \
    /path/to/vipsania.sif \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3 --finetune
```

If your cluster provides Docker instead of Singularity, use
`--gpus "device=${CUDA_VISIBLE_DEVICES}"` so the container respects the
SLURM-assigned GPU rather than claiming all GPUs on the node:

```bash
#!/bin/bash
#SBATCH --job-name=vipsania
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

sudo docker run --rm --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    -v "$SCRATCH/data:/data" \
    -v "$SCRATCH/models:/cache/vipsania/models" \
    katharinahoff/vipsania \
    vipsania annotate Fungi /data/genome.fa -o /data/annotation.gff3 --finetune
```

---

## GPU troubleshooting inside a container

If Vipsania reports that no GPU was found, the container runtime is not
exposing the GPU correctly.  Check:

- **Docker**: confirm `sudo docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`
  prints the GPU list.  If it fails, re-check the NVIDIA Container Toolkit
  installation.
- **Singularity**: confirm `singularity exec --nv vipsania.sif nvidia-smi`
  prints the GPU list.  If it fails, the host driver may be too old; update it
  or contact your system administrator.
- **LD_LIBRARY_PATH**: the container entrypoint already sets this to
  TensorFlow's bundled CUDA libraries.  If you launch the container's Python
  directly (bypassing the entrypoint) you may need to set it manually — see
  [troubleshooting.md](troubleshooting.md).
