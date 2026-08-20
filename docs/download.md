# Downloading models

A pretrained model is a folder named after its model ID that holds a `config.json` and the file
`latest_checkpoint.weights.h5`. Annotation fetches the model it needs by itself the first time it
is used, so in the normal case there is nothing to do — this page is for deciding where the models
are kept and for machines that cannot reach the internet when they annotate.

## Where models are stored

Downloaded models are placed in `$XDG_CACHE_HOME/vipsania/models`, or in `~/.cache/vipsania/models`
if that variable is not set. They are shared between all your environments and survive reinstalling
Vipsania. Set `VIPSANIA_CACHE` to store them somewhere else, which is worth doing on clusters with
a small home quota:

    $ export VIPSANIA_CACHE=/scratch/$USER/vipsania

Annotations started at the same time may all want the same model, as happens when a whole array job
is submitted at once. That is safe — they will not get in each other's way — but every one of them
downloads the model, so fetch it once beforehand instead.

## Fetching a model in advance

    $ vipsania download 58hsuobw

Without further options this fills exactly the cache the annotation reads from, so the annotation
command afterwards stays the same. That is what to run on a login node before submitting jobs to
compute nodes without internet access.

To collect models somewhere else, for example in a directory shared by a whole group, give a target
directory and pass it to the annotation as `--model_dir`:

    $ vipsania download 58hsuobw faeijtmk --dir /shared/vipsania_models
    $ vipsania annotate 58hsuobw genome.fa -o annotation.gff3 --model_dir /shared/vipsania_models

| option          | meaning                                                                |
| --------------- | ---------------------------------------------------------------------- |
| `-d`, `--dir`   | directory to download into, defaults to the cache described above       |
| `--force`       | download again even if the model is already there                       |

Several models can be given at once, by clade name or by ID. A model that is already present is
reported and skipped. `versions.json` is placed in the target directory as well, so that clade
names also work later with `--model_dir`.

## Getting a model there by hand

The download needs no special tooling: a model folder is two files. Copying the folder of an
already downloaded model to another machine works, as does building it yourself from a training run
— that is what `--model_dir` is for.
