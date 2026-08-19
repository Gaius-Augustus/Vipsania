# Vipsania documentation

The [main README](/README.md) is the quick path: install Vipsania, pick a model, annotate a genome.
These pages go into detail.

| page | contents |
| ---- | -------- |
| [annotate.md](/docs/annotate.md) | options of `vipsania annotate`, notes on repeat masking |
| [training.md](/docs/training.md) | the configuration format, preparing data, running and resuming a training, non-standard genetic codes |
| [download.md](/docs/download.md) | where models are stored, fetching them in advance |
| [troubleshooting.md](/docs/troubleshooting.md) | what to check when no GPU is found |

Two tables record which species were involved in each model:

| file | contents |
| ---- | -------- |
| [training_species.tsv](/docs/training_species.tsv) | the species every model was trained on, with NCBI assembly accessions |
| [test_species.tsv](/docs/test_species.tsv) | the test species, with precision, sensitivity and F1 at base, exon and locus level, with and without finetuning |

## The command

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
