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

Unknown keys are rejected, so a typo in a configuration fails immediately instead of being silently
ignored.

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

The base configurations list model, dataset and training details. Use the
[configs/train.json](/configs/train.json) file to describe *your* data and pass it with `-oc`,
which overrides the base configuration entry by entry.

```json
{
    "dataset": {
        "train_paths": [
            "/path/to/genomes/species_A.fa",
            "/path/to/genomes/species_B.fa"
        ],
        "indexed_files": true,
        "indexed_window_size": 3200000,
        "indexed_windows_at_once": 4
    }
}
```

    $ vipsania train configs/base_10M.json -oc configs/train.json

**The one entry you have to fill in is `train_paths`.** It accepts three forms, which can be mixed:

- plain paths to FASTA files,
- paths containing `*`, which are expanded as globs, e.g. `"/path/to/genomes/*/*.fa"`,
- a single path to a text file that lists one FASTA file per line.

Vipsania samples random windows from the listed files instead of loading them into memory, so
genomes of any size can be used as they are and do not have to be split up beforehand. Files are
drawn roughly in proportion to their size, capped so that a single very large genome cannot
dominate the training. Sequences shorter than the context length `dataset.T` are skipped.

Two entries control how those windows are read. `dataset.indexed_window_size` is the number of
nucleotides fetched from a file in one go, and `dataset.indexed_windows_at_once` is how many such
windows are read in parallel.

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

## Running out of GPU memory

The shipped configurations are sized for the GPUs the published models were trained on, and the
larger ones do not fit on a smaller card. A run that dies with `ResourceExhaustedError` needs a
smaller batch, and two other entries have to follow so that the training itself stays the same:

- lower `dataset.B`,
- raise `trainer.gradient_accumulation_steps` so that `B x gradient_accumulation_steps` stays 64,
- set `trainer.train_steps` to `100 x gradient_accumulation_steps`.

The batch is then assembled from more, smaller pieces: the number of sequences per epoch and the
number of weight updates stay what they were, only the memory needed at one moment goes down. The
shipped configurations follow the same rule with `B = 8` and 8 accumulation steps. On a 24 GB card,
`B = 2` works for the 10M model and `B = 1` for the 25M one.

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
[configs/train.json](/configs/train.json) configuration and edit them. Each entry is a list of
`[pattern, probability]` pairs whose probabilities should sum to one, and `N` stands for any
nucleotide, so `NGT` is the usual GT donor together with the preceding base and `AGN` the AG
acceptor.

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

`-oc` takes a whole file of overrides, as above. Single values can also be given on the command
line with `-o`, using dots to address nested keys, which is convenient for sweeps and for one-off
changes:

    $ vipsania train configs/base_25M.json -oc configs/train.json -o dataset.B=4

Both can be combined; where they set the same entry, the file wins.

## Continuing from a published model

A pretrained model can be trained further on genomes of your choice. This is what `vipsania
annotate --finetune` does for a single genome. Here, we extend this to finetuning on multiple
genomes. Fill the FASTA files into [configs/resume.json](/configs/resume.json) and name the model to
continue from, by clade or by model ID:

    $ vipsania train Fungi --resume --fork finetuned_fungi -oc configs/resume.json

`--resume` picks the training up from the weights of that model instead of starting from scratch,
and the override brings the settings that suit a continuation: a low learning rate, no warmup or
decay, and no schedules. `--fork` copies the model into `finetuned_fungi` first and trains there,
so the model you started from stays exactly as it was and can be used or forked again at any time.
It is downloaded on the way if you do not have it yet. The folder has to be new, and `--fork` only
works together with `--resume`.

The result is a model folder like any other, which `vipsania annotate` uses through `--model_dir`:

    $ vipsania annotate finetuned_fungi genome.fa -o annotation.gff3 --model_dir .

Inside it, the weights you started from are kept as `epoch_0000.weights.h5` and their configuration
as `config_epoch_0000.json`. The new `latest_checkpoint.weights.h5` is about four times the size of
the one you downloaded, because it carries the optimizer state along; that is expected.

Without `--fork` the run writes into the folder of the model itself, which is what you want when
continuing your own training, but not when the model came from a download: a run that stops before
finishing its first epoch would leave that folder without a `latest_checkpoint.weights.h5`.

If the run stops with `ResourceExhaustedError`, adjust `B`, `gradient_accumulation_steps` and
`train_steps` in the same file, following the rule in
[Running out of GPU memory](#running-out-of-gpu-memory).

## Further options

| option          | meaning                                                                      |
| --------------- | ---------------------------------------------------------------------------- |
| `--checkpoints` | directory in which the run folder is created, defaults to `./checkpoints`     |
| `--fork`        | with `--resume`: continue in this new folder, leaving the model untouched    |
| `--mirrored`    | train on multiple GPUs with a mirrored strategy; requires `--nojit`           |
| `--nojit`       | disable JIT compilation of the model, which is slower but easier to debug     |
| `--summary`     | build the model, print its summary and exit                                   |
