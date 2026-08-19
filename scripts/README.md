# Vipsania scripts

Installing Vipsania provides the `vipsania` command with three subcommands:

- `vipsania annotate` — annotate a genome with a trained model.
- `vipsania train` — train a new model from scratch or resume a prior training run.
- `vipsania download` — fetch a pretrained model ahead of time, which annotation otherwise does
  by itself.

The scripts in this directory are thin wrappers around exactly those subcommands, so that a cloned
repository can be used without relying on the installed command. All three forms are equivalent:

    $ vipsania train configs/base_25M.json
    $ python -m vipsania train configs/base_25M.json
    $ python scripts/train.py configs/base_25M.json

Every command prints all available options with `--help`. The [main README](/README.md) gives a
quick overview of annotating a genome; this document lists the full options of the commands and
describes training in detail.

## Annotation

    $ vipsania annotate model_id genome.fa -o annotation.gff3 --finetune

The [main README](/README.md) explains the model ID, the output formats and why finetuning is
recommended for every annotation. Below are useful options of the command.

### Common options

| option               | meaning                                                              |
| -------------------- | -------------------------------------------------------------------- |
| `-o`, `--output`     | output file; the suffix selects the format                            |
| `-T`, `--context`    | genome context length in nucleotides, defaults to `200_000`           |
| `-B`, `--batch_size` | batch size; inferred from the available GPU memory by default         |
| `-i`, `--include`    | only annotate the named sequences                                     |
| `-e`, `--exclude`    | skip the named sequences                                              |
| `--model_dir`        | directory containing the model folder; skips the automatic download   |
| `--weights`          | file name of the weights inside the model folder                      |
| `--keep_seqnames`    | do not strip sequence names at the first whitespace character         |
| `--protein`          | also write the protein sequences of all predicted genes to this file  |
| `--coding`           | also write the coding sequences of all predicted genes to this file   |

### Finetuning options

| option              | meaning                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `--finetune`        | train on the genome before annotating it                                  |
| `--finetune_epochs` | number of epochs to finetune, defaults to `10`                            |
| `--finetune_lr`     | learning rate used for finetuning, defaults to `1e-4`                     |
| `--finetune_B`      | batch size on the GPU; inferred from the available GPU memory by default  |
| `--drop_repeats`    | only finetune on sequences with low repeat content                        |

`--finetune_B` only affects how much GPU memory is used; the effective batch size is always 64,
reached through gradient accumulation. If `--drop_repeats` is not given, it is chosen from the size
of the input genome.

On strongly repetitive genomes, sequences below the allowed repeat content can be so rare that the
dataloader spends most of its time searching for them, which stretches a finetuning run from hours
into a day. Vipsania watches how many sequences the filter discards, warns when the search starts
to dominate, and then raises the allowed repeat content in steps of 10 percentage points until the
sampling runs freely again. Setting `--drop_repeats 0` switches the filter off entirely.

### Performance

| option          | meaning                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `--group_limit` | limits the size of the sequence groups annotated at once; reduce this for a slower but more memory-friendly annotation |
| `--delta`       | a larger delta helps with genomes made up of many small sequences              |
| `-p`            | degree of parallelization of the HMM; derived from the context length by default |
| `--nojit`       | do not compile the model with JIT                                              |

## Training

Training a Vipsania model is **unsupervised**: the only input is a set of FASTA files. No
annotation, no labels and no evidence of any kind are needed, which means that any genome you have
access to can be used as training data.

A training run is fully described by a single JSON configuration file with three sections:

| section   | contents                                                                       |
| --------- | ------------------------------------------------------------------------------ |
| `model`   | architecture of the network, including the HMM layer                            |
| `dataset` | which FASTA files to train on, context length, batch size and masking behaviour |
| `trainer` | epochs, steps, learning rate schedule and everything else about optimization    |

Every key corresponds to a field of `VipsaniaConfig`, `DatasetConfig` and `TrainerConfig` in the
Python package, and unknown keys are rejected, so a typo fails immediately instead of being
silently ignored.

### Model configurations

Ready-to-use configurations live in [configs](/configs). They differ only in model size. We
recommend the 2M or 10M model for single species training, based on the species genome size. The
25M variant is the one used for all released checkpoints.

| configuration                          | layers | hidden size |
| -------------------------------------- | ------ | ----------- |
| [base_2M.json](/configs/base_2M.json)   | 6      | 150         |
| [base_10M.json](/configs/base_10M.json) | 10     | 256         |
| [base_25M.json](/configs/base_25M.json) | 16     | 320         |

All three use the same block layout — a bidirectional linear recurrent unit, a gated feed-forward
network and a single HMM block placed in the middle of the stack — and the same data and
optimization settings.

### Adapting a configuration to your data

**The one entry you have to change is `dataset.train_paths`.** It ships empty and has to be filled
with the FASTA files you want to train on:

```json
"dataset": {
    "T": 20000,
    "B": 8,
    "train_paths": [
        "/path/to/genomes/species_A.fa",
        "/path/to/genomes/species_B.fa"
    ]
}
```

`train_paths` accepts three forms, which can be mixed:

- plain paths to FASTA files,
- paths containing `*`, which are expanded as globs, e.g. `"/path/to/genomes/*/*.fa"`,
- a single path to a text file that lists one FASTA file per line.

Vipsania samples random windows from the listed files instead of loading them into memory, so
genomes of any size can be used as they are and do not have to be split up beforehand. Files are
drawn roughly in proportion to their size, capped so that a single very large genome cannot
dominate the training. Sequences shorter than the context length `dataset.T` are skipped.

Two entries control how those windows are read. `dataset.indexed_window_size` is the number of
nucleotides fetched from a file in one go, and `dataset.indexed_windows_at_once` is how many such
windows are read in parallel. Every window is cut into `indexed_window_size / T` training
sequences, and the sequences of all open windows are shuffled together. With the defaults of
`6_400_000` and `1` and a context length of `20_000`, one window yields 320 training sequences that
are shuffled among themselves.

Since each window is drawn from an independently sampled file, `indexed_windows_at_once` is what
mixes genomes: at `1`, consecutive batches come from a single species until its window is used up,
while raising it interleaves that many files at a time. **When training on many FASTA files, raise
it** — at the cost of holding that many windows in memory at once.

| entry                                 | meaning                                                   |
| ------------------------------------- | --------------------------------------------------------- |
| `dataset.T`                           | context length in nucleotides used during training         |
| `dataset.B`                           | batch size; lower this first if you run out of GPU memory  |
| `dataset.indexed_windows_at_once`     | number of windows, and therefore files, read in parallel; defaults to `1` |
| `dataset.indexed_window_size`         | nucleotides read from a file in one go; defaults to `6_400_000` |
| `trainer.gradient_accumulation_steps` | raise this to keep the effective batch size when lowering `B` |
| `trainer.epochs`                      | number of epochs                                           |
| `trainer.train_steps`                 | number of batches per epoch                                |

### Running a training

    $ vipsania train configs/base_25M.json

Everything the run produces is written to a new folder below `./checkpoints`: the resolved
configuration, the checkpoints and a summary of the last finished epoch. Use `--checkpoints` to
place that folder somewhere else. The name of the run folder is the **model ID**, and it is exactly
what you pass to the annotation command afterwards.

    $ vipsania annotate <model_id> genome.fa -o annotation.gff3 --finetune

To see the model that a configuration describes without starting a training, use `--summary`.

    $ vipsania train configs/base_25M.json --summary

### Logging to Weights & Biases

Configure your [Weights & Biases](https://wandb.ai) API token and pass your entity and project.

    $ vipsania train configs/base_25M.json --online my_entity/my_project

With online logging enabled, the run folder is named after the Weights & Biases run ID, so a run
can always be traced back from the model ID.

### Overriding configuration values

Instead of editing a configuration file, single values can be overridden on the command line with
`-o`, using dots to address nested keys. This is convenient for sweeps and for restarting a run
with a small change.

    $ vipsania train configs/base_25M.json -o dataset.B=4 -o trainer.lr=1e-4

A JSON file with a set of overrides can be applied with `-oc/--overrideconfig`, which is
particularly useful together with `--resume`.

### Resuming a run

    $ vipsania train <model_id> --resume

Passing a model ID instead of a configuration file continues from the latest checkpoint of that
run. The previous checkpoint and configuration are kept, and the epoch counter continues where it
stopped.

### Further options

| option          | meaning                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| `--checkpoints` | directory in which the run folder is created, defaults to `./checkpoints`     |
| `--mirrored`    | train on multiple GPUs with a mirrored strategy; requires `--nojit`           |
| `--nojit`       | disable JIT compilation of the model, which is slower but easier to debug     |
| `--summary`     | build the model, print its summary and exit                                   |

## Troubleshooting

### Vipsania reports that no GPU was found

`vipsania annotate` and `vipsania train` print the devices they are going to use when they
start. If no GPU is reported
although the machine has one, TensorFlow was unable to load its CUDA libraries and silently fell
back to the CPU. Check it directly:

    $ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

An empty list, together with a message such as `Cannot dlopen some GPU libraries`, means that the
CUDA libraries installed alongside TensorFlow are not on the library search path, or that an older
CUDA installation somewhere on the system is being loaded instead. Rerun with
`TF_CPP_MIN_LOG_LEVEL=0` to see which library failed. Pointing the loader at the libraries that came
with TensorFlow usually resolves it:

    $ export LD_LIBRARY_PATH=$(python -c "import site, glob, os; print(':'.join(sorted(glob.glob(os.path.join(site.getsitepackages()[0], 'nvidia', '*', 'lib')))))")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

To make this permanent inside a conda environment, put that line into a script in
`$CONDA_PREFIX/etc/conda/activate.d/`.

Vipsania requires `tensorflow<2.20` for the same reason: the 2.21.0 wheel no longer finds the
cuSOLVER library that pip installs next to it, and therefore registers no GPU at all.
