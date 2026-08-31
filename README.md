# Vipsania

**Vipsania is an unsupervised deep learning *ab-initio* gene finder.**

[Preprint at bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.26.747235v1)

Unlike other deep learning gene finders, Vipsania is never shown a reference annotation. It is
trained purely on raw genomic sequences with a masked language modelling objective: nucleotides are
hidden and the model learns to predict them from their context. Gene structure emerges as the
representation a hidden Markov model inside the network needs in order to explain the sequence, so
no curated gene set, no related-species annotation and no protein evidence are required at any
point.

For you, this means:

- **Annotate any eukaryotic genome from a FASTA file alone.** No extrinsic evidence, no reference
  annotation, no gene-structure training data of any kind.
- **Adapt the model to your species on the fly.** Because training needs nothing but sequence, the
  genome you want to annotate is itself valid training data. A short finetuning run on your target
  genome directly before annotating gives the best results — see
  [Finetuning](#finetuning-recommended).
- **Get standard output.** Annotations are written as GenePred, GFF3 or GTF.

## Installation

Vipsania requires `python>=3.12`. It is recommended to install into a clean virtual environment
(e.g. using `conda`, `pyenv` or `uv`).

    $ python -m pip install vipsania

This installs the `vipsania` command, which is everything you need to annotate a genome. For
training, use the [config files](/configs) provided and follow the instructions [here](#training).

Alternatively, you can clone the repository, install Vipsania from source and use the
[scripts](/scripts) for annotation and training.

    $ git clone https://github.com/gaius-augustus/vipsania
    $ python -m pip install -e vipsania/

Installing Vipsania pulls in [bricks2marble](https://github.com/gaius-augustus/bricks2marble) and
[hidten](https://github.com/gaius-augustus/hidten), which in turn install TensorFlow with CUDA
support. A GPU is strongly recommended; annotating a large genome on CPU only is possible, but
slow.

## Annotating a genome

Annotation is done with `vipsania annotate`. It needs a trained model and a FASTA file of the
genome you want to annotate.

    $ vipsania annotate <model_id> genome.fa -o annotation.gff3 --finetune

The same command is available as `python -m vipsania annotate ...`, and from a cloned repository as
`python scripts/annotate.py ...`.

The first argument names the model to use, either by the **clade** it was trained for or by its
**model ID**, both listed below. The first time a model is used, Vipsania downloads it (about 100
MB) and keeps it, so every later annotation with it starts immediately.

    $ vipsania annotate Insecta genome.fa -o annotation.gff3 --finetune

A clade name gives you the model we recommend for that clade at the moment. Should a clade get a
better model later, `vipsania download <clade>` picks it up; annotating on its own keeps using the
model it already has. A model ID always refers to that one model.

A model you trained yourself is given in the same way: its ID is the name of its run folder, which
Vipsania looks for in the directories around the working directory. Set `--model_dir` to use a
different directory that holds the model folder directly.

Progress is printed to the terminal while the genome is annotated, together with a log file next to
the output.

### Choosing a model

Vipsania is trained per clade. Use the model of the clade your species belongs to, choosing the
most specific one available.

| Clade | Model ID | Training species | Test species | Average locus F1 | Average locus F1 (pretrained) |
| --- | --- | ---: | ---: | ---: | ---: |
| Alveolata | sd2zcj7u | 30 | 6 | 0.613 | 0.582 |
| Amoebozoa | ezhpj2qm | 15 | 7 | 0.437 | 0.416 |
| Arthropoda | ihe1jk30 | 200 | 6 | 0.207 | 0.217 |
| Chlorophyta | faeijtmk | 26 | 6 | 0.579 | 0.558 |
| Cnidaria | b5vtieo0 | 82 | 10 | 0.453 | 0.406 |
| Discoba | gcra9d9y | 20 | 4 | 0.696 | 0.696 |
| Echinodermata | sx9zjl7p | 52 | 8 | 0.444 | 0.440 |
| Fungi | fh1kg88z | 200 | 7 | 0.660 | 0.645 |
| Insecta | 58hsuobw | 200 | 9 | 0.537 | 0.474 |
| Nematoda | zspca2rb | 73 | 8 | 0.539 | 0.517 |
| Porifera | 7bjiexcu | 75 | 6 | 0.480 | 0.416 |
| Rhodophyta | 6kmw3wme | 20 | 5 | 0.405 | 0.301 |
| Spiralia | v5ej8oyt | 200 | 6 | 0.374 | 0.262 |
| Stramenopiles | r6p9z9jw | 32 | 6 | 0.426 | 0.413 |
| Streptophyta | j9m0cdmk | 200 | 13 | 0.602 | 0.579 |
| Tunicata | hcehc7ff | 34 | 6 | 0.523 | 0.494 |
| Vertebrata | etb1go6q | 200 | 13 | 0.406 | 0.327 |
| | | | | | |
| *none of the above* | cg6grhms | 33 | 7 | 0.436 | 0.416 |

Use the last entry for eukaryotes that belong to none of the clades above — it was trained on
exactly such species. It has no clade name of its own; pass `Other` or its model ID.

The two F1 columns are the average locus F1 over the test species of that clade, measured against
their reference annotations: the first with `--finetune`, the second with the pretrained model
alone. Finetuning helps in almost every clade, which is why we recommend it.

Whichever model you pick, finetuning it on your genome adapts it to your species — see
[Finetuning](#finetuning-recommended).

The species every model was trained on are listed in
[docs/training_species.tsv](/docs/training_species.tsv), one row per species with its NCBI
assembly accession. Since Vipsania never learns from annotations, the quality of a species'
reference annotation, or whether it has one at all, leaves no trace in the model, and unlike for a
supervised gene finder, finding your species or a close relative in the list is not an argument
against that model.

The test species behind the two F1 columns are in
[docs/test_species.tsv](/docs/test_species.tsv), with precision, sensitivity and F1 at base,
exon and locus level for each of them, once with and once without finetuning.

### Where models are stored

The first time a model ID is used it is downloaded into `~/.cache/vipsania/models` and read from
there ever after. See [docs/download.md](/docs/download.md) for changing that location, and for
fetching models in advance on machines that have no internet access when they annotate.

### Output formats

The output format is chosen by the file suffix of `-o/--output`:

| suffix           | format                                      |
| ---------------- | ------------------------------------------- |
| `.gp`            | GenePred                                    |
| `.gff3` / `.gff` | GFF3                                        |
| `.gtf`           | GTF                                         |

If `-o` is omitted, the annotation is written next to the input FASTA as `vipsania_[model_id].gff`,
named after the model that produced it even when you asked for it by clade. The protein and coding
sequences of the predicted genes can be written out at the same time with `--protein proteins.fa`
and `--coding coding.fa`.

### Finetuning (recommended)

    $ vipsania annotate <model_id> genome.fa -o annotation.gff3 --finetune

`--finetune` trains the model on the genome you are about to annotate before predicting anything.
Since Vipsania is unsupervised, this needs nothing except the FASTA file that you already have. The
model adapts to the codon usage, repeat landscape and intron statistics of your species, which
consistently improves the annotation.

The finetuned checkpoint is saved next to the output file, so you can reuse it for further
annotations of the same genome or closely related species without finetuning again. Use the
`--model_dir` argument in these cases.

All options of the annotation are listed in [docs/annotate.md](/docs/annotate.md), or with
`vipsania annotate --help`.

## Training

Vipsania models are trained with `vipsania train` on a set of local FASTA files. Ready-to-use
configurations for the three model sizes are in [configs](/configs); rather than editing one, you
list the FASTA files you want to train on in [configs/train.json](/configs/train.json) and pass it
as an override:

    $ vipsania train configs/base_10M.json -oc configs/train.json

Training runs can be logged to [Weights & Biases](https://wandb.ai) with `--online entity/project`.
Everything a run produces — configuration, checkpoints and metrics — is written to a folder below
`./checkpoints`, named after the run. That name is exactly the model ID that `vipsania annotate`
expects.

A published model can also be trained further on species of your choice, which extends the
`--finetune` option of `vipsania annotate` to a set of genomes. See
[docs/training.md](/docs/training.md) for that, for the configuration format, and for what to
change when a run does not fit into GPU memory.

## Built on

Vipsania is built on two libraries developed in our group:

- [bricks2marble](https://github.com/gaius-augustus/bricks2marble) — efficient handling of
  nucleotide sequences and genome annotations, plus the pre- and postprocessing around genome
  annotation.
- [hidten](https://github.com/gaius-augustus/hidten-docs) — hidden Markov models as differentiable,
  highly parallel layers inside deep learning models.
