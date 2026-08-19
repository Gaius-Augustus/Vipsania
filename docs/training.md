# Training a model

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

## Model configurations

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

## Adapting a configuration to your data

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

## Species with a non-standard genetic code

The HMM does not assume a genetic code, it is told one. Four entries of the `hmm` section spell out
which codons start and end a gene and which patterns flank an intron. They are not written in the
shipped configurations because every model so far uses the standard code, which is what these
fields default to:

```json
"hmm": {
    "start_codons": [["ATG", 1.0]],
    "stop_codons": [["TAG", 0.34], ["TAA", 0.33], ["TGA", 0.33]],
    "intron_begin_pattern": [["NGT", 0.99], ["NGC", 0.01]],
    "intron_end_pattern": [["AGN", 1.0]]
}
```

To train on species that deviate, copy these four lines into the `hmm` section of your
configuration and edit them. Each entry is a list of `[pattern, probability]` pairs whose
probabilities should sum to one, and `N` stands for any nucleotide, so `NGT` is the usual GT donor
together with the preceding base and `AGN` the AG acceptor.

Ciliates, for instance, read `TAA` and `TAG` as glutamine and stop only at `TGA`:

```json
"stop_codons": [["TGA", 1.0]]
```

The same mechanism covers non-canonical splice sites: extend `intron_begin_pattern` or
`intron_end_pattern` with the additional motifs and give each a share of the probability.

## Running a training

    $ vipsania train configs/base_25M.json

Everything the run produces is written to a new folder below `./checkpoints`: the resolved
configuration, the checkpoints and a summary of the last finished epoch. Use `--checkpoints` to
place that folder somewhere else. The name of the run folder is the **model ID**, and it is exactly
what you pass to the annotation command afterwards.

    $ vipsania annotate <model_id> genome.fa -o annotation.gff3 --finetune

To see the model that a configuration describes without starting a training, use `--summary`.

    $ vipsania train configs/base_25M.json --summary

## Logging to Weights & Biases

Configure your [Weights & Biases](https://wandb.ai) API token and pass your entity and project.

    $ vipsania train configs/base_25M.json --online my_entity/my_project

With online logging enabled, the run folder is named after the Weights & Biases run ID, so a run
can always be traced back from the model ID.

## Overriding configuration values

Instead of editing a configuration file, single values can be overridden on the command line with
`-o`, using dots to address nested keys. This is convenient for sweeps and for restarting a run
with a small change.

    $ vipsania train configs/base_25M.json -o dataset.B=4 -o trainer.lr=1e-4

A JSON file with a set of overrides can be applied with `-oc/--overrideconfig`, which is
particularly useful together with `--resume`.

## Resuming a run

    $ vipsania train <model_id> --resume

Passing a model ID instead of a configuration file continues from the latest checkpoint of that
run. The previous checkpoint and configuration are kept, and the epoch counter continues where it
stopped.

## Further options

| option          | meaning                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| `--checkpoints` | directory in which the run folder is created, defaults to `./checkpoints`     |
| `--mirrored`    | train on multiple GPUs with a mirrored strategy; requires `--nojit`           |
| `--nojit`       | disable JIT compilation of the model, which is slower but easier to debug     |
| `--summary`     | build the model, print its summary and exit                                   |
