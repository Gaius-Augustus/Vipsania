from pathlib import Path
from typing import TYPE_CHECKING, Literal

import bricks2marble as b2m
import numpy as np
import tensorflow as tf

from .mask import token_masking, token_masking_sequentially

if TYPE_CHECKING:
    from .watch import RepeatSamplingWatcher


def fasta_to_tensors(
    fasta: Path | str | b2m.struct.Fasta | b2m.struct.Sequence,
    T: int | None = None,
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    repeats_output: Literal["separate", "omit"] = "omit",
    force_cpu: bool = False,
    dtype: tf.DType = tf.float32,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Reads a Fasta file at the given path (or uses the one already
    supplied) and returns tensors of input and output data.
    """
    if isinstance(fasta, (Path, str)):
        fasta = b2m.io.load_fasta(Path(fasta).expanduser(), T=T)
    y = fasta.nuc.copy()
    y[y == 4] = -1
    if repeats_output == "omit":
        y[y > 4] = y[y > 4] - 5
    else:
        y[y > 4] = y[y > 4] - 1
    if force_cpu:
        with tf.device("CPU:0"):
            x_, y_ = (
                tf.convert_to_tensor(
                    fasta.one_hot(repeats=repeats_input, N=N_token),
                    dtype=dtype,
                ),
                tf.convert_to_tensor(y, dtype=dtype),
            )
        return x_, y_
    return (
        tf.convert_to_tensor(
            fasta.one_hot(repeats=repeats_input, N=N_token),
            dtype=dtype,
        ),
        tf.convert_to_tensor(y, dtype=dtype),
    )


def sequentially_masked(
    x: tf.Tensor | tf.data.Dataset,
    p: float,
) -> tf.data.Dataset:
    every = int(1 / p)
    if not isinstance(x, tf.data.Dataset):
        x = tf.data.Dataset.from_tensor_slices(x).batch(1)
    return tf.data.Dataset.from_tensor_slices(tf.range(every)).flat_map(
        lambda k: x.map(
            lambda tensor: token_masking_sequentially(tensor, every, k)
        )
    )


def drop_N_sequences(
    dataset: tf.data.Dataset,
    threshold: float,
) -> tf.data.Dataset:
    """Drop sequences that have more than `T * threshold` N tokens from
    the dataset, where `T` is the chunk length.
    """
    def _drop_N(x, y):
        with tf.device("CPU:0"):
            flag = tf.reduce_sum(
                tf.cast(x[..., 4] == 1, tf.float32)
            ) <= threshold * tf.cast(tf.shape(x)[0], tf.float32)
        return flag
    return dataset.filter(_drop_N)


def drop_repeats_sequences(
    dataset: tf.data.Dataset,
    threshold: float,
    watcher: "RepeatSamplingWatcher | None" = None,
) -> tf.data.Dataset:
    """Drop sequences that have more than `T * threshold` repeats from
    the dataset, where `T` is the chunk length. Repeats are assumed to
    be an extra track in the sixth component of the input.

    With a `watcher`, the threshold is taken from it instead, which
    allows raising it while the dataset is read on genomes where too few
    sequences pass the filter.
    """
    limit = threshold if watcher is None else watcher.threshold

    def _drop_repeats(x, y):
        with tf.device("CPU:0"):
            flag = tf.reduce_sum(
                tf.cast(x[..., 5] == 1, tf.float32)
            ) <= limit * tf.cast(tf.shape(x)[0], tf.float32)
        if watcher is not None:
            return watcher.observe(flag)
        return flag
    return dataset.filter(_drop_repeats)


def masked(
    dataset: tf.data.Dataset,
    masking: float,
    token: float | None = None,
    same: float | None = None,
    false: float | None = None,
    seed: int | None = None,
    weight_repeat_masked: float = 0,
) -> tf.data.Dataset:
    """Creates a masked version of the given dataset containing
    nucleotide sequences.
    """
    return dataset.map(
        lambda x_, y_: token_masking(x_, y_,
            masking=masking,
            token=token,
            same=same,
            false=false,
            seed=seed,
            weight_repeat_masked=weight_repeat_masked,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def expand_path(path: Path) -> list[Path]:
    """Expands path components that contain an asterisk to any possible
    existing paths.
    """
    total_paths = [Path()]
    for component in path.expanduser().parts:
        new_paths = []
        for i in range(len(total_paths)):
            if "*" in component:
                new_paths.extend([p for p in total_paths[i].glob(component)])
            else:
                new_paths.append(total_paths[i] / component)
        total_paths = new_paths
    return total_paths


def mutate_once(sequence: np.ndarray) -> np.ndarray:
    """For a given sequence of nucleotides ``{0, 1, 2, 3}`` of shape
    ``(T, )``, returns a sequence ``mutations`` of shape
    ``(T, T, 3)``, where ``mutations[:, t, i]`` is the same as the input
    sequence, except for position ``t`` being mutated into nucleotide
    ``i`` (for all ``i != sequence[t]``).
    """
    T = sequence.shape[0]
    all_bases = np.arange(4)
    alt_bases = np.stack([
        all_bases[all_bases != b] for b in sequence
    ], axis=0)

    mutations = np.tile(sequence[:, np.newaxis, np.newaxis], (1, T, 3))
    idx = np.arange(T)
    mutations[idx, idx, :] = alt_bases
    return mutations


def mutate_once_ds(
    sequence: np.ndarray,
    region: tuple[int, int] | None = None,
) -> tf.data.Dataset:
    T = sequence.shape[0] if region is None else (region[1] - region[0])
    start, end = region if region is not None else (0, T)
    mutations = mutate_once(sequence[start:end])
    mutations = np.transpose(mutations, (1, 2, 0)).reshape(T*3, T)
    if region is not None:
        mutations = np.concatenate((
            np.tile(np.expand_dims(sequence[:start], axis=0), (T*3, 1)),
            mutations,
            np.tile(np.expand_dims(sequence[end:], axis=0), (T*3, 1))
        ), axis=1)
    mutations = np.concatenate((sequence[np.newaxis, :], mutations), axis=0)
    mutations = b2m.struct.fasta.one_hot(mutations)
    mutations = np.concatenate(
        (mutations, np.zeros(mutations.shape[:-1] + (1, ))),
        axis=-1,
    )
    ds = tf.data.Dataset.from_tensor_slices(mutations)
    return ds
