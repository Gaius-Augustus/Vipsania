# Vipsania in the Paludamentum pipeline

[Paludamentum](https://github.com/Gaius-Augustus/Paludamentum) is the Nextflow pipeline around the
gene finders of the Gaius-Augustus family. It prepares extrinsic evidence (proteins, RNA-Seq,
Iso-Seq), derives high-confidence genes from it, and merges them with the *ab initio* prediction
of the gene finder. Vipsania is one of its gene finders. Paludamentum runs the Vipsania container
image; nothing in this repository depends on Paludamentum.

## Getting Paludamentum

Vipsania is a git submodule of Paludamentum, which pins the Vipsania version whose container image
the pipeline runs:

    $ git clone --recursive https://github.com/Gaius-Augustus/Paludamentum
    $ cd Paludamentum && pip install .

Nextflow, Java and Singularity/Apptainer are needed on the machine that launches the pipeline.

## Running Vipsania through the pipeline

    $ paludamentum --genefinder vipsania --nf_config slurm_generic \
          --genome genome.fa --model Fungi --proteins proteins.faa

or with a params file that has `vipsania: {run: true, model: Fungi}`:

    $ paludamentum --params_yaml params.yaml --nf_config slurm_generic

Without evidence the pipeline only parallelizes `vipsania annotate` over the chunks of the genome.
The results are `vipsania_evidence.gff3`, `vipsania_evidence_proteins.fa` and
`vipsania_ab_initio.gff3` in the output directory.

## What the pipeline does with Vipsania

- The genome is split into chunks that are annotated in parallel with `vipsania annotate`.
- Finetuning is off by default. With `--finetune` (or `vipsania.finetune: true`), the pipeline
  runs one `vipsania annotate --finetune` task on the whole genome instead, because Vipsania cannot
  finetune without annotating. The checkpoint is published with the intermediate results.
- The model is downloaded on the submitting host, or taken from `--model_dir` on clusters without
  internet access (see [download.md](/docs/download.md)).
- The Vipsania processes run in `docker://gaiusaugustus/vipsania:<version>` through Singularity,
  with `--nv`. The GPU notes in [container.md](/docs/container.md) apply.

All Vipsania parameters of the pipeline (`vipsania.model`, `model_dir`, `finetune`,
`finetune_epochs`, `batch_size`, `context`, `max_parallel`, ...) are documented in the Paludamentum
repository (`docs/vipsania.md`).
