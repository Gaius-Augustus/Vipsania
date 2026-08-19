from collections.abc import Sequence
from typing import Literal, overload

import bricks2marble as b2m
import numpy as np
import tensorflow as tf

from ..data.mask import pseudo_mask
from ..data.util import mutate_once_ds, sequentially_masked


@tf.function(jit_compile=True)
def model_call(model, x: tf.Tensor) -> tf.Tensor:
    return model(x)


@overload
def predict_dataset(
    model: tf.keras.Model | tf.keras.Layer,
    ds: tf.data.Dataset,
    hooks: None = ...,
    return_batched: bool = ...,
    jit_compile: bool = ...,
) -> tf.Tensor:
    ...
@overload
def predict_dataset(
    model: tf.keras.Model | tf.keras.Layer,
    ds: tf.data.Dataset,
    hooks: Sequence[tuple[tf.keras.Layer, str]] = ...,
    return_batched: bool = ...,
    jit_compile: bool = ...,
) -> tuple[tf.Tensor, ...]:
    ...
def predict_dataset(
    model: tf.keras.Model | tf.keras.Layer,
    ds: tf.data.Dataset,
    hooks: Sequence[tuple[tf.keras.Layer, str]] | None = None,
    return_batched: bool = False,
    jit_compile: bool = False,
) -> tuple[tf.Tensor, ...] | tf.Tensor:
    output = []
    if hooks is not None:
        for hook in hooks:
            hook[0].attach_hooks()
    for k, tensor in enumerate(ds):  # type: ignore
        if jit_compile:
            out = model_call(model, tensor)
        else:
            out = model(tensor)
        if k == 0:
            output.append(
                tf.zeros((0, ) + tuple(out.shape[1:]), dtype=out.dtype)
            )
        output[0] = tf.concat(
            (output[0], out),
            axis=0,
        )
        if hooks is not None:
            for i, hook in enumerate(hooks):
                out = hook[0].hooks[hook[1]]
                if k == 0:
                    output.append(tf.zeros(
                        (0, ) + tuple(out.shape[1:]),
                        dtype=out.dtype,
                    ))
                output[i+1] = tf.concat((output[i+1], out), axis=0)
    if hooks is not None:
        for hook in hooks:
            hook[0].release_hooks()
    result = []
    for i in range(1 + (len(hooks) if hooks is not None else 0)):
        result.append(
            tf.reshape(output[i], (-1, )+tuple(output[i].shape[2:]))
            if not return_batched else output[i]
        )
    return tuple(result) if len(result) > 1 else result[0]


@overload
def predict_sequence(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence | b2m.struct.Fasta,
    B: int = ...,
    hooks: None = ...,
    return_batched: bool = ...,
    N_token: Literal["track", "uniform"] = ...,
    repeats_input: Literal["track", "expand", "omit"] = ...,
    masked: bool = ...,
    jit_compile: bool = ...,
) -> tf.Tensor:
    ...
@overload
def predict_sequence(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence | b2m.struct.Fasta,
    B: int = ...,
    hooks: Sequence[tuple[tf.keras.Layer, str]] = ...,
    return_batched: bool = ...,
    N_token: Literal["track", "uniform"] = ...,
    repeats_input: Literal["track", "expand", "omit"] = ...,
    masked: bool = ...,
    jit_compile: bool = ...,
) -> tuple[tf.Tensor, ...]:
    ...
def predict_sequence(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence | b2m.struct.Fasta,
    B: int = 32,
    hooks: Sequence[tuple[tf.keras.Layer, str]] | None = None,
    return_batched: bool = False,
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    masked: bool = True,
    jit_compile: bool = False,
) -> tuple[tf.Tensor, ...] | tf.Tensor:
    data = tf.convert_to_tensor(
        sequence.one_hot(repeats=repeats_input, N=N_token),
        dtype=tf.float32,
    )
    if masked: data = pseudo_mask(data)
    ds = tf.data.Dataset.from_tensor_slices(data).batch(B)
    return predict_dataset(
        model,
        ds,
        hooks=hooks,
        return_batched=return_batched,
        jit_compile=jit_compile,
    )


@overload
def predict(
    model: tf.keras.Model | tf.keras.Layer,
    x: tf.Tensor,
    B: int = ...,
    T: int | None = ...,
    hooks: None = ...,
    return_batched: bool = ...,
) -> tf.Tensor:
    ...
@overload
def predict(
    model: tf.keras.Model | tf.keras.Layer,
    x: tf.Tensor,
    B: int = ...,
    T: int | None = ...,
    hooks: Sequence[tuple[tf.keras.Layer, str]] = ...,
    return_batched: bool = ...,
) -> tuple[tf.Tensor, ...]:
    ...
def predict(
    model: tf.keras.Model | tf.keras.Layer,
    x: tf.Tensor,
    B: int = 1,
    T: int | None = None,
    hooks: Sequence[tuple[tf.keras.Layer, str]] | None = None,
    return_batched: bool = False,
) -> tuple[tf.Tensor, ...] | tf.Tensor:
    length, D = tf.shape(x)[0], tf.shape(x)[1]  # type: ignore
    if T is None:
        T = int(length)
    if length % T > 0:
        raise ValueError("T has to divide sequence length")
    ds = tf.data.Dataset.from_tensor_slices(tf.reshape(x, (-1, T, D))).batch(B)
    return predict_dataset(
        model,
        ds,
        hooks=hooks,
        return_batched=return_batched,
    )


@overload
def predict_interleave(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence,
    p_mask: float,
    B: int = ...,
    hooks: None = ...,
    N_token: Literal["track", "uniform"] = ...,
    repeats_input: Literal["track", "expand", "omit"] = ...,
) -> tf.Tensor:
    ...
@overload
def predict_interleave(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence,
    p_mask: float,
    B: int = ...,
    hooks: Sequence[tuple[tf.keras.Layer, str]] = ...,
    N_token: Literal["track", "uniform"] = ...,
    repeats_input: Literal["track", "expand", "omit"] = ...,
) -> tuple[tf.Tensor, ...]:
    ...
def predict_interleave(
    model: tf.keras.Model | tf.keras.Layer,
    sequence: b2m.struct.Sequence,
    p_mask: float,
    B: int = 1,
    hooks: Sequence[tuple[tf.keras.Layer, str]] | None = None,
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    jit_compile: bool = False,
) -> tuple[tf.Tensor, ...] | tf.Tensor:
    """Uses the given model to predict each position in ``x`` exactly
    once. To do this, the tensor is transformed into a dataset with
    batch size ``B`` and length ``T``, where each ``int(1/p_mask)``-th
    token is masked. Only this masked token will be predicted and
    finally concatenated in the right order to yield the total result.

    Args:
        model (tf.keras.Model or Layer): The model used to predict the
            output.
        sequence (Sequence): A bricks2marble nucleotide sequence.
        p_mask (float): The percentage of tokens to be masked in the
            sequence at once. Masked tokens are spaced equidistantly.
        B (int, optional): Number of sequences to be fed at once into
            the model. Can be used to speed up predictions. Defaults to
            1.

    Returns:
        Tensor: A tensor with the predictions of shape ``(N*T, D)``
            where ``N`` is the number of chunks of length ``T`` in the
            sequence and ``D`` is the size of the model output.
    """
    with tf.device("CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(
                sequence.one_hot(repeats=repeats_input, N=N_token),
                dtype=tf.float32,
            ),
        ).batch(B)
        ds = sequentially_masked(ds, p=p_mask)

    a = [0 for _ in range(sequence.size)]
    every = int(1 / p_mask)
    T_per_offset = [len(a[k::every]) for k in range(every)]
    output_per_offset = []
    indices = tf.zeros((0, 1), dtype=tf.int64)
    offset = 0
    t = 0
    if hooks is not None:
        for hook in hooks:
            hook[0].attach_hooks()
    for k, (mask, tensor) in enumerate(ds):  # type: ignore
        if k > 0 and (tf.shape(output_per_offset[0][-1])[0]  # type: ignore
                        >= T_per_offset[offset]):
            for i in range((len(hooks) if hooks is not None else 0) + 1):
                output_per_offset[i].append(tf.zeros(
                    (0, ) + tuple(output_per_offset[i][-1].shape[1:]),
                    dtype=tf.float32,
                ))
            offset += 1
            t = 0
        if jit_compile:
            output = model_call(model, tensor)
        else:
            output = model(tensor)
        output = tf.cast(output, tf.float32)
        if k == 0:
            output_per_offset.append([
                tf.zeros((0, )+tuple(output.shape[2:]), dtype=tf.float32)
            ])
        output_per_offset[0][-1] = tf.concat(
            (output_per_offset[0][-1], tf.boolean_mask(output, mask)),
            axis=0,
        )
        if hooks is not None:
            for i, hook in enumerate(hooks):
                output = hook[0].hooks[hook[1]]
                if k == 0:
                    output_per_offset.append([tf.zeros(
                        (0, )+tuple(output.shape[2:]),
                        dtype=tf.float32,
                    )])
                output_per_offset[i+1][-1] = tf.concat((
                    output_per_offset[i+1][-1], tf.boolean_mask(output, mask)
                ), axis=0)
        where = tf.where(tf.reshape(mask, [-1]))
        indices = tf.concat(
            (indices, tf.cast(t, tf.int64) + where),
            axis=0,
        )
        t += tf.shape(output)[0] * tf.shape(output)[1]  # type: ignore
    if hooks is not None:
        for hook in hooks:
            hook[0].release_hooks()
    result = []
    for i in range((len(hooks) if hooks is not None else 0) + 1):
        output_per_offset[i] = tf.concat(  # type: ignore
            output_per_offset[i],
            axis=0,
        )
        result.append(
            tf.tensor_scatter_nd_update(
                tf.zeros(
                    (sequence.N*sequence.T, )
                        + tuple(output_per_offset[i].shape[1:]),
                    dtype=output.dtype,
                ),
                indices,
                output_per_offset[i],
            )
        )
    return tuple(result) if hooks is not None else result[0]


def dependencies_scores(
    model: tf.keras.Model,
    sequence: b2m.struct.Sequence,
    mutate_region: tuple[int, int] | None = None,
    B: int = 1,
    forbid_repeats: bool = False,
    reduce_max: bool = True,
    jit_compile: bool = True,
) -> np.ndarray:
    """Returns the pairwise nucleotide dependency score that measures
    the dependency of one query nucleotide in a sequence to another
    target nucleotide by mutating the query.

    From paper:

        "Nucleotide dependency analysis of genomic language models
         detects functional elements." ~ da Silva et. al (2025)

    Returns:
        np.ndarray: Array ``score`` of shape ``(T, T)``, where
        ``score[t, s]`` is the maximum log odds ratio that is inferred
        from the model when mutating a query nucleotide at position
        ``t`` into all variants.
    """
    seq = sequence.flat
    if not forbid_repeats:
        seq[seq > 4] = seq[seq > 4] - 5
    if np.any(seq == 4):
        raise ValueError(
            "Sequences with N "
            f"{'or repeat masked positions ' if forbid_repeats else ''}"
            "are not allowed."
        )
    ds = mutate_once_ds(seq, region=mutate_region)
    ds = ds.batch(B)
    prediction = predict_dataset(
        model,
        ds,
        return_batched=True,
        jit_compile=jit_compile,
    )
    if mutate_region is None:
        T = seq.shape[0]
        prediction = tf.nn.softmax(prediction, axis=-1).numpy()  # type: ignore
    else:
        start, end = mutate_region
        T = end - start
        prediction = tf.nn.softmax(
            prediction[:, start:end],
            axis=-1,
        ).numpy()  # type: ignore

    base_prediction = prediction[0].reshape(T, 4)
    prediction = prediction[1:].reshape(T, 3, T, 4)

    log_odds_ratio = np.log2(
        prediction / base_prediction[np.newaxis, np.newaxis, :, :]
    )
    if reduce_max:
        dependency_score = np.max(log_odds_ratio, axis=(1, 3))
        np.fill_diagonal(dependency_score, 0)
        return dependency_score
    return log_odds_ratio
