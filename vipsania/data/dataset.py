from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Sequence, overload

import bricks2marble as b2m
import numpy as np
import tensorflow as tf
from pydantic import BaseModel

from .util import (drop_N_sequences, drop_repeats_sequences, expand_path,
                   fasta_to_tensors, masked)

if TYPE_CHECKING:
    from .watch import RepeatSamplingWatcher


class DatasetConfig(BaseModel):

    T: int
    B: int | None = None

    N_token: Literal["track", "uniform"] = "track"
    repeats_input: Literal["track", "expand", "omit"] = "track"
    repeats_output: Literal["separate", "omit"] = "omit"
    repeats_loss_weight: float = 0

    masking: float | None = 0.15
    token: float | None = 1.0
    same: float | None = None
    false: float | None = None

    shuffle: int | None = 10_000
    drop_N_threshold: float = 0.0
    drop_repeats_threshold: float = 0.0
    seed: int | None = None
    repeat_dataset: bool = True
    tile_targets: int = 1

    train_paths: Sequence[Path | str]
    train_files_at_once: int = 1
    validation_paths: Sequence[Path | str] | None = None
    validation_files_at_once: int = 1

    indexed_files: bool = False
    indexed_window_size: int = 6_400_000
    indexed_windows_at_once: int = 1

    @property
    def input_dim(self) -> int:
        """The input token size with masking."""
        match self.repeats_input:
            case "track": dim = 6
            case "expand": dim = 9
            case "omit": dim = 5
        if self.N_token == "uniform":
            dim -= 1
        return dim + 1

    model_config = {"extra": "forbid", "frozen": True}


def parallel_files(
    files: Path | str | Sequence[Path | str],
    at_once: int,
    config: DatasetConfig,
    watcher: "RepeatSamplingWatcher | None" = None,
) -> tf.data.Dataset:
    if isinstance(files, (Path, str)):
        with open(files, "r") as f:
            files = f.read().split("\n")
    files_new = []
    for file in files:
        files_new.extend(expand_path(Path(file)))
    files = files_new

    at_once = min(len(files), at_once)
    config = config

    def _path_to_dataset(path_tensor: tf.Tensor) -> tf.data.Dataset:
        def _loader(p: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            x, y = tf.py_function(
                func=lambda p: fasta_to_tensors(
                    p.numpy().decode(),
                    T=config.T,
                    N_token=config.N_token,
                    repeats_input=config.repeats_input,
                    repeats_output=config.repeats_output,
                    force_cpu=True,
                    dtype=tf.float32,
                ),
                inp=[p],
                Tout=[tf.float32, tf.float32],
            )  # type: ignore
            x.set_shape((None, config.T, config.input_dim - 1))
            y.set_shape((None, config.T))
            return x, y

        ds = tf.data.Dataset.from_tensors(path_tensor).map(_loader).unbatch()
        if config.drop_N_threshold > 0:
            ds = drop_N_sequences(ds, config.drop_N_threshold)
        if config.drop_repeats_threshold > 0:
            ds = drop_repeats_sequences(
                ds, config.drop_repeats_threshold, watcher=watcher,
            )
        if config.masking is not None:
            ds = masked(
                ds,
                masking=config.masking,
                token=config.token,
                same=config.same,
                false=config.false,
                seed=config.seed,
                weight_repeat_masked=config.repeats_loss_weight,
            )
        if config.tile_targets > 1:
            def tile_targets(x, y, sample_weights=None):
                if sample_weights is None:
                    return x, tuple(y for _ in range(config.tile_targets))
                return x, tuple(y for _ in range(config.tile_targets)), tuple(
                    sample_weights for _ in range(config.tile_targets)
                )
            ds = ds.map(tile_targets)
        return ds

    ds = tf.data.Dataset.list_files([str(p) for p in files], shuffle=True)
    ds = ds.interleave(
        _path_to_dataset,
        cycle_length=at_once,
        num_parallel_calls=at_once,
        deterministic=False,
    )
    if config.repeat_dataset: ds = ds.repeat()
    if config.shuffle is not None: ds = ds.shuffle(buffer_size=config.shuffle)
    if config.B is not None: ds = ds.batch(config.B)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def select_from_indexed_files(
    files: Path | str | Sequence[Path | str],
    config: DatasetConfig,
    watcher: "RepeatSamplingWatcher | None" = None,
    extra: Sequence[tuple[
        str,
        tf.TensorSpec,
        Callable[[Path, str, tuple[int, int]], tf.Tensor],
    ]] | None = None,
) -> tf.data.Dataset:
    """Takes a number of paths to `.fa.gz` files compressed with `bgzip`
    and creates a dataset based on the given configuration.

    Additional target tensors can be loaded with the `extra` argument.
    Each target has to be specified using:

        - name (str): Name of the file with the target data. Has to be
            in the same directory as the corresponding fasta file.
        - spec (tf.TensorSpec): The specs (dtype and shape) of the new
            target values.
        - generator_function (callable): A callable that takes a path to
            such a target file, a sequence name and a range in that
            sequence to return the tensor of that range.
    """
    if isinstance(files, (Path, str)):
        with open(files, "r") as f:
            files = f.read().split("\n")
    files_new = []
    for file in files:
        files_new.extend(expand_path(Path(file)))
    files = files_new

    if extra is None:
        extra = []

    fastas = [b2m.io.indexed_fasta(file) for file in files]
    extras = [[f.parent/s for f in files] for s, _, _ in extra]

    seqs = [[s for s in f.sequence_names()] for f in fastas]
    seq_sizes = [[
        fastas[g].length(name) for name in names
    ] for g, names in enumerate(seqs)]
    genome_sizes = [sum(s) for s in seq_sizes]
    p_seq = []
    for sizes in seq_sizes:
        effective = [max(0, size - config.T + 1) for size in sizes]
        total = sum(effective)
        p_seq.append([e / total for e in effective])

    genome_median = np.median(genome_sizes)
    p_genome = [
        (g if g <= genome_median else genome_median)
        for g in genome_sizes
    ]
    total = sum(p_genome)
    p_genome = [g/total for g in p_genome]

    def sample_window():
        """Yields (genome_idx, seq_idx, start, actual_window_size)."""
        while True:
            g = np.random.choice(len(p_genome), p=p_genome)
            s = np.random.choice(len(p_seq[g]), p=p_seq[g])
            seq_size = seq_sizes[g][s]
            window = min(config.indexed_window_size, seq_size)
            if window < config.T: continue
            b = np.random.choice(seq_size - window + 1)
            yield g, s, b, window

    def load_window(g, s, b, window):
        """Load a large window and return all T-slices as a dataset."""
        g, s, b, window = g.numpy(), s.numpy(), b.numpy(), window.numpy()
        raw = fastas[g].fetch(seqs[g][s], (b, b + window))
        xy = fasta_to_tensors(
            raw.resample(config.T),
            T=config.T,
            N_token=config.N_token,
            repeats_input=config.repeats_input,
            repeats_output=config.repeats_output,
            force_cpu=True,
            dtype=tf.float32,
        )
        N, T = tf.unstack(tf.shape(xy[1]))
        for e in range(len(extra)):
            extra_tensor = extra[e][2](extras[e][g], seqs[g][s], (b, b+window))
            extra_tensor = tf.reshape(extra_tensor, (N, T))
            xy = xy + (extra_tensor, )
        if len(extra) > 0:
            xy = (xy[0], ) + (tuple(xy[1:]), )
        return xy

    ds_signature = (
        tf.TensorSpec(
            shape=(None, config.T, config.input_dim-1),
            dtype=tf.float32,
        ),
    )
    if len(extra) == 0:
        ds_signature += (
            tf.TensorSpec(shape=(None, config.T), dtype=tf.float32),
        )
    else:
        ds_signature += (
            (tf.TensorSpec(shape=(None, config.T), dtype=tf.float32), )
            + tuple(extra[i][1] for i in range(len(extra))),
        )

    def load_window_wrapper(g, s, b, window):
        outputs = tf.py_function(
            lambda g, s, b, w: tf.nest.flatten(load_window(g, s, b, w)),
            inp=[g, s, b, window],
            Tout=[spec.dtype for spec in tf.nest.flatten(ds_signature)],
        )
        for tensor, spec in zip(
            tf.nest.flatten(outputs),
            tf.nest.flatten(ds_signature),
        ):
            tensor.set_shape(spec.shape)
        repacked = tf.nest.pack_sequence_as(
            ds_signature,
            tf.nest.flatten(outputs),
        )
        return tf.data.Dataset.from_tensor_slices(repacked)

    outer = tf.data.Dataset.from_generator(
        sample_window,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),  # g
            tf.TensorSpec(shape=(), dtype=tf.int32),  # s
            # the start is int64: it is turned into a byte offset in the
            # fasta file, which passes 2^31 in genomes larger than 2 GB
            tf.TensorSpec(shape=(), dtype=tf.int64),  # b
            tf.TensorSpec(shape=(), dtype=tf.int32),  # window
        ),
    )

    ds = outer.interleave(
        load_window_wrapper,
        cycle_length=config.indexed_windows_at_once,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if config.drop_N_threshold > 0:
        ds = drop_N_sequences(ds, config.drop_N_threshold)
    if config.drop_repeats_threshold > 0:
        ds = drop_repeats_sequences(
            ds, config.drop_repeats_threshold, watcher=watcher,
        )
    if config.masking is not None:
        ds = masked(
            ds,
            masking=config.masking,
            token=config.token,
            same=config.same,
            false=config.false,
            seed=config.seed,
            weight_repeat_masked=config.repeats_loss_weight,
        )
    if config.tile_targets > 1:
        def tile_targets(x, y, sample_weights=None):
            if sample_weights is None:
                return x, tuple(y for _ in range(config.tile_targets))
            return x, tuple(y for _ in range(config.tile_targets)), tuple(
                sample_weights for _ in range(config.tile_targets)
            )
        ds = ds.map(tile_targets)
    ds = ds.shuffle(buffer_size=(
        config.indexed_windows_at_once * (config.indexed_window_size//config.T)
    ))
    if config.B is not None: ds = ds.batch(config.B)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


@overload
def train_val_split(
    fasta: Path | str | b2m.struct.Fasta,
    config: DatasetConfig,
    val_size: float = ...,
    return_count: Literal[False] = ...,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    ...
@overload
def train_val_split(
    fasta: Path | str | b2m.struct.Fasta,
    config: DatasetConfig,
    val_size: float = ...,
    return_count: Literal[True] = ...,
) -> tuple[tf.data.Dataset, int, tf.data.Dataset, int]:
    ...
def train_val_split(
    fasta: Path | str | b2m.struct.Fasta,
    config: DatasetConfig,
    val_size: float = 0.2,
    return_count: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset] | tuple[
    tf.data.Dataset, int, tf.data.Dataset, int
]:
    """Returns two :class:`tf.data.Dataset` objects by randomly dividing
    the total number of sequence chunks from the given Fasta file into
    train and validation examples.
    """
    x, y = fasta_to_tensors(
        fasta,
        T=config.T,
        force_cpu=True,
    )
    with tf.device("CPU:0"):
        train, val = tf.keras.utils.split_dataset(
            tf.data.Dataset.from_tensor_slices((x, y)),
            left_size=1-val_size,
            shuffle=config.shuffle is not None,
            seed=config.seed,
        )

    train: tf.data.Dataset
    val: tf.data.Dataset
    n_train, n_val = train.cardinality().numpy(), val.cardinality().numpy()
    if config.masking is not None:
        train = train.repeat()
        train = masked(
            train,
            masking=config.masking,
            token=config.token,
            same=config.same,
            false=config.false,
            seed=config.seed,
            weight_repeat_masked=config.repeats_loss_weight,
        )
        val = val.repeat()
        val = masked(
            val,
            masking=config.masking,
            token=config.token,
            same=config.same,
            false=config.false,
            seed=config.seed,
            weight_repeat_masked=config.repeats_loss_weight,
        )
    if config.B is not None:
        train = train.batch(config.B)
        val = val.batch(config.B)
    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    if return_count:
        return train, n_train, val, n_val
    return train, val
