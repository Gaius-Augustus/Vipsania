# Vipsania documentation

The [main README](/README.md) is the quick path: install Vipsania, pick a model, annotate a genome.
These pages go into detail.

| page | contents |
| ---- | -------- |
| [annotate.md](/docs/annotate.md) | options of `vipsania annotate`, notes on repeat masking |
| [training.md](/docs/training.md) | training on your own genomes, continuing from a published model, GPU memory, non-standard genetic codes |
| [download.md](/docs/download.md) | where models are stored, fetching them in advance |
| [troubleshooting.md](/docs/troubleshooting.md) | what to check when no GPU is found |
| [example](/docs/example) | a chromosome and the annotation Vipsania produced for it, to test an installation |

Two tables record which species were involved in each model:

| file | contents |
| ---- | -------- |
| [training_species.tsv](/docs/training_species.tsv) | the species every model was trained on, with NCBI assembly accessions |
| [test_species.tsv](/docs/test_species.tsv) | the test species, with precision, sensitivity and F1 at base, exon and locus level, with and without finetuning |

## Example

The folder [example](/docs/example) holds chromosome 7 of *Aspergillus fumigatus* (NCBI assembly
`GCF_000002655.1`), 2 Mb of sequence, together with the annotation Vipsania produced for it. Run it
to see whether your installation works and gives what it should:

    $ cd docs/example
    $ vipsania annotate fh1kg88z aspergillus_fumigatus_chr7.fa -o my_annotation.gff3

`fh1kg88z` is the Fungi model, which is downloaded on the way. Writing to a file of your own with
`-o` allows you to compare the two. Ours contains 693 genes and took 55 seconds;
[vipsania_fh1kg88z.log](/docs/example/vipsania_fh1kg88z.log) records that run from end to end.

Against the chromosome's reference annotation of that assembly, available from NCBI, this
prediction reaches a locus sensitivity of 57% and a precision of 53.5%, measured with `gffcompare
--strict-match -T -e 3` (v0.12.6). Finetuning was not used here.

## CLI

Installing Vipsania provides the `vipsania` command with three subcommands:

- `vipsania annotate` — annotate a genome with a trained model.
- `vipsania train` — train a new model from scratch or resume a prior training run.
- `vipsania download` — fetch a pretrained model ahead of time, which annotation otherwise does
  by itself.

The scripts in [scripts](/scripts) are thin wrappers around exactly those subcommands, so that a
cloned repository can be used without relying on the installed command. All three forms are
equivalent:

    $ vipsania train configs/base_25M.json
    $ python -m vipsania train configs/base_25M.json
    $ python scripts/train.py configs/base_25M.json

Every command prints all available options with `--help`.
