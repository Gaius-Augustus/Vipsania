# Embedding sequences

    $ vipsania embed <model_id> sequences.fa -o embeddings.h5

Instead of an annotation, `vipsania embed` writes the residual stream of a model for every
nucleotide of a fasta file. The output is an HDF5 file with one dataset of shape
`(length, d_hidden)` per sequence, named after it. The model, layer and stripe are stored as
attributes of the file. Without `-o` it is written next to the input as `vipsania_[model_id].h5`.

```python
import h5py

with h5py.File("embeddings.h5") as f:
    print(dict(f.attrs))          # model, layer, stripe and context length
    x = f["chr1"][:]              # shape (length of chr1, d_hidden)
```

## Common options

| option               | meaning                                                              |
| -------------------- | -------------------------------------------------------------------- |
| `-o`, `--output`     | output HDF5 file                                                      |
| `-l`, `--layer`      | one or more layers the residual stream is taken from, defaults to `7` |
| `-s`, `--stripe`     | stripe within each layer, defaults to `2`                             |
| `-T`, `--context`    | genome context length in nucleotides, defaults to `200_000`           |
| `-B`, `--batch_size` | batch size; inferred from the available GPU memory by default         |
| `--exact`            | embed every sequence at its own length, without padding               |
| `--dtype`            | `float32` (default) or `float16`, which halves the file size          |
| `-i`, `--include`    | only embed the named sequences                                        |
| `-e`, `--exclude`    | skip the named sequences                                              |
| `--model_dir`        | directory containing the model folder; skips the automatic download   |
| `--weights`          | file name of the weights inside the model folder                      |
| `--keep_seqnames`    | do not strip sequence names at the first whitespace character         |

## Layers and stripes

Every layer of a Vipsania model is a series of stripes, such as an LRU, a feed forward network or
an HMM. `-l` and `-s` choose where the residual stream is read, both counted from 0; the default is
the output of the HMM in the 25M models. A pair that does not exist is reported together with the
valid ones.

Several layers can be embedded in one run, with one stripe per layer or a single stripe for all of
them. Each pair is written to its own file, named after `-o` with `_l[layer]s[stripe]` appended,
and the run costs as much as the deepest pair alone:

    $ vipsania embed <model_id> sequences.fa -o embeddings.h5 -l 7 15 -s 2 1

## Exact embeddings

Sequences shorter than a chunk are padded with `N`, which changes their embeddings. `--exact`
embeds every sequence at its own length instead. This trades speed for exactness: every length
would need its own compilation, so the model runs without JIT. It is meant for sets of differently
sized sequences that each fit into one chunk, such as regions cut out of a genome.

## Performance

| option          | meaning                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| `--group_limit` | limits the size of the sequence groups embedded at once; reduce this for a slower but more memory-friendly run |
| `--delta`       | a larger delta pads short sequences less, at the cost of more groups           |
| `-p`            | degree of parallelization of the HMM; derived from the context length by default |
| `--nojit`       | do not compile the model with JIT                                              |
