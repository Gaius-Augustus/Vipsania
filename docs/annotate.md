# Annotating a genome

    $ vipsania annotate model_id genome.fa -o annotation.gff3 --finetune

## Common options

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

## Finetuning options

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

## Repeat-masked genomes

Vipsania reads the soft masking of a genome — lowercase nucleotides — as a repeat track and feeds
it to the model alongside the sequence. Good masking helps, but masking that is too aggressive or
too sparse misleads the model, and the quality of a repeat annotation is often hard to judge.

If you are unsure about yours, do not guess: strip the masking and let finetuning learn the repeat
structure of your genome from the sequence itself.

    $ awk '/^>/ {print; next} {print toupper($0)}' genome.fa > genome_unmasked.fa
    $ vipsania annotate model_id genome_unmasked.fa -o annotation.gff3 --finetune

## Performance

| option          | meaning                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `--group_limit` | limits the size of the sequence groups annotated at once; reduce this for a slower but more memory-friendly annotation |
| `--delta`       | a larger delta helps with genomes made up of many small sequences              |
| `-p`            | degree of parallelization of the HMM; derived from the context length by default |
| `--nojit`       | do not compile the model with JIT                                              |
