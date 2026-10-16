from collections.abc import Container, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import bricks2marble as b2m
import numpy as np
import tensorflow as tf
from bricks2marble.struct.fasta import one_hot

from .model.base import Vipsania


def _as_list(value, name: str) -> list:
    if isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        return [value]
    if len(value) == 0:
        raise ValueError(f"No {name} given.")
    return list(value)


def _resolve_positions(
    model: Vipsania,
    layers: list[int],
    stripes: list[int],
) -> list[tuple[int, int]]:
    """Pairs of layer and stripe as non-negative indices, with a single
    layer or stripe repeated to the length of the other list.
    """
    if len(layers) != len(stripes) and 1 not in (len(layers), len(stripes)):
        raise ValueError(
            f"Got {len(layers)} layers but {len(stripes)} stripes; give "
            "one stripe per layer, or a single layer or stripe for all."
        )
    n = max(len(layers), len(stripes))
    layers = layers * (n // len(layers))
    stripes = stripes * (n // len(stripes))

    positions = []
    for layer, stripe in zip(layers, stripes):
        try:
            strp = model.stripes[layer][stripe]
        except IndexError as e:
            valid = [
                (lid, sid) for lid in range(len(model.stripes))
                for sid in range(len(model.stripes[lid]))
            ]
            raise IndexError(
                f"Layer and stripe index ({layer}, {stripe}) out of bounds. "
                f"Valid pairs are: {valid}"
            ) from e
        positions.append((strp.layer_id, strp.stripe_id))
    if len(set(positions)) < len(positions):
        raise ValueError(f"Layer and stripe pairs repeat: {positions}")
    return positions


def embed_fasta(
    model: Vipsania,
    fasta: Path | str,
    output: Path | str | Sequence[Path | str],
    T: int,
    parallel: int,
    layer: int | Sequence[int] = 7,
    stripe: int | Sequence[int] = 2,
    B: int = 1,
    exact: bool = False,
    T_delta: float = 0.1,
    group_limit: int = 1_000_000_000,
    include: Container[str] | None = None,
    exclude: Container[str] | None = None,
    split_seqnames: bool = True,
    dtype: Literal["float32", "float16"] = "float32",
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    jit_compile: bool = False,
    attrs: Mapping[str, str | int | float] | None = None,
) -> None:
    """Writes the residual stream of a Vipsania model after the given
    layer and stripe for every nucleotide of every sequence in a fasta
    file to an HDF5 file.

    The output file holds one dataset of shape ``(L, d_hidden)`` per
    sequence of length ``L``, named after the sequence. Sequences are
    grouped by length and split into chunks as in the annotation, so
    ``T``, ``parallel``, ``T_delta`` and ``group_limit`` have the same
    meaning as in :func:`vipsania.annotate_genome`. Chunks are embedded
    independently of each other, and the last chunk of a sequence is
    padded with ``N``.

    That padding is not free: the embeddings of a sequence change with
    the number of ``N`` appended to it, and by how much depends on that
    number rather than on the length of the sequence. A few dozen
    padded positions move them by about a percent, a few hundred by
    more than a tenth of their norm. For a set of sequences that each
    fit into one chunk, ``exact`` therefore embeds every length on its
    own and leaves no padding at all.

    Several layers and stripes are embedded in a single pass: ``layer``,
    ``stripe`` and ``output`` may be sequences, one output file per pair
    of layer and stripe. The model then runs up to the deepest of them
    only, and all others are read on the way, so the cost is that of the
    deepest pair alone. Without jit, every intermediate result of that
    pass is kept until the batch is done, so large batches need
    correspondingly more GPU memory.

    Args:
        model (Vipsania): The model to take the embeddings from.
        fasta (Path | str): Fasta file with the sequences to embed.
        output (Path | str | Sequence): The HDF5 file to write, or one
            file per pair of layer and stripe; overwritten if they
            exist.
        T (int): Maximal chunk length fed into the model at once.
        parallel (int): Degree of parallelization of the HMM; has to
            divide ``T``.
        layer (int | Sequence[int], optional): Index of the layer the
            residual stream is taken from, starting at 0. Defaults to 7.
        stripe (int | Sequence[int], optional): Index of the stripe
            within that layer, starting at 0. Defaults to 2, the HMM
            stripe in the 25M models. A single layer or stripe is used
            for every entry of the other.
        B (int, optional): Batch size for chunks of length ``T``.
            Groups of shorter chunks use a proportionally larger batch.
            Defaults to 1.
        exact (bool, optional): Embed each sequence at its own length,
            so that nothing is padded. Sequences longer than ``T`` are
            still split into chunks of length ``T``. The degree of
            parallelization is lowered per group to a divisor of its
            length, which does not change the result. Defaults to
            False.
        dtype (str, optional): Floating point type the embeddings are
            stored with. Defaults to "float32".
        attrs (Mapping, optional): Additional attributes written to the
            root of every output file.
    """
    import h5py

    positions = _resolve_positions(
        model, _as_list(layer, "layer"), _as_list(stripe, "stripe"),
    )
    outputs = _as_list(output, "output")
    if len(outputs) != len(positions):
        raise ValueError(
            f"Got {len(outputs)} output files for {len(positions)} pairs of "
            "layer and stripe."
        )

    deepest = max(positions)
    names = [f"layer_{lid}_stripe_{sid}" for lid, sid in positions]
    hooked = len(positions) > 1

    def forward(x: tf.Tensor) -> list[tf.Tensor]:
        if not hooked:
            return [model(x)]
        model.clear_hooks()
        model.attach_hooks()
        y = model(x)
        ys = [
            y if position == deepest else model.hooks[name]
            for position, name in zip(positions, names)
        ]
        model.release_hooks()
        model.clear_hooks()
        return ys

    # compiled, only the requested intermediate results leave the graph
    call = tf.function(forward, jit_compile=True) if jit_compile else forward

    model.toggle_inference(
        "embedding", layer=deepest[0], stripe=deepest[1], enable=True,
    )
    print(
        f"Embedding {fasta} with the residual stream after " + " and ".join(
            f"layer {lid}, stripe {sid}" for lid, sid in positions
        ),
        flush=True,
    )

    try:
        with ExitStack() as stack:
            files = [stack.enter_context(h5py.File(o, "w")) for o in outputs]
            for file, (lid, sid) in zip(files, positions):
                file.attrs.update(
                    {"layer": lid, "stripe": sid, "T": T} | dict(attrs or {})
                )

            n_sequences = 0
            for group in b2m.io.iterate_sequences(
                fasta,
                T_max=T,
                # a group of one length only, so that nothing is padded
                delta=1.0 if exact else T_delta,
                T_factors=[1] if exact else [parallel],
                group_size_limit=group_limit,
                include=include,
                exclude=exclude,
                split_name=split_seqnames,
            ):
                T_group = group.T
                nuc = group.nuc
                B_group = max(1, min(B * T // T_group, nuc.shape[0]))
                if exact:
                    # the HMM splits a chunk into segments of equal length,
                    # which is exact for every divisor of the chunk length
                    p = min(parallel, T_group)
                    while T_group % p: p -= 1
                    model.set_options(parallel=p)
                    done = n_sequences + len(group)
                    if n_sequences // 500 < done // 500:
                        print(f"{done} sequences", flush=True)
                else:
                    print(
                        f"{len(group)} sequence{'s' if len(group) > 1 else ''}"
                        f" in {nuc.shape[0]} chunks of length {T_group}, "
                        f"batch size {B_group}",
                        flush=True,
                    )

                # chunk k of the group belongs to sequence owner[k] and
                # starts at position offset[k] of it
                owner = np.repeat(np.arange(len(group)), [s.N for s in group])
                offset = np.concatenate(
                    [np.arange(s.N) * T_group for s in group]
                )
                datasets = [[None] * len(group) for _ in positions]

                for start in range(0, nuc.shape[0], B_group):
                    batch = nuc[start:start+B_group]
                    n = batch.shape[0]
                    # the last batch is padded, so that no new graph needs
                    # to be compiled for it
                    if n < B_group:
                        batch = np.concatenate((batch, np.full(
                            (B_group-n, T_group), -1, dtype=batch.dtype,
                        )))
                    x = one_hot(batch, repeats=repeats_input, N=N_token)
                    # the input of the model carries an additional flag
                    # for masked positions, which is never set here
                    x = np.concatenate(
                        (x, np.zeros(x.shape[:-1] + (1, ), dtype=x.dtype)),
                        axis=-1,
                    )
                    ys = call(tf.convert_to_tensor(x))

                    for file, sets, y in zip(files, datasets, ys):
                        y = y.numpy()[:n].astype(dtype)
                        for k in range(n):
                            i = owner[start+k]
                            seq = group[int(i)]
                            if sets[i] is None:
                                sets[i] = file.create_dataset(
                                    seq.name,
                                    shape=(seq.size, y.shape[-1]),
                                    dtype=dtype,
                                )
                            o = offset[start+k]
                            L = min(T_group, seq.size - o)
                            sets[i][o:o+L] = y[k, :L]

                n_sequences += len(group)
    finally:
        model.toggle_inference("embedding", enable=False)

    print(
        f"Wrote embeddings of {n_sequences} sequences to "
        + ", ".join(str(o) for o in outputs)
    )
