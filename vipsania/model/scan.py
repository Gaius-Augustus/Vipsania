import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_util


def _get_static_predicate(pred):
    """Shortend version of a function with the same name from
    `tensorflow_probability.python.internal.prefer_static`.
    """
    pred_value = pred
    if tf.is_tensor(pred):
        pred_value = tf.get_static_value(tf.convert_to_tensor(pred))
        if pred_value is None and isinstance(pred, tf.Tensor):
            pred_value = tensor_util.try_evaluate_constant(pred)

    if pred_value in (0, 1, True, False):
        pred_value = bool(pred_value)
    elif pred_value is not None:
        raise TypeError(
            "`pred` must be a Tensor, or a Python bool, or 1 or 0. "
            f"Found instead: {pred}"
        )
    return pred_value


def _static_cond(pred, true_fn, false_fn):
    """Shortend version of
    `tensorflow_probability.python.internal.prefer_static.cond`.
    """
    pred_value = _get_static_predicate(pred)
    if pred_value is not None:
        if pred_value:
            return true_fn()
        else:
            return false_fn()
    return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn)


def _slice_along_axis(x, start=0, stop=None, step=1, axis=0):
    """Shortend version of a function with the same name from
    `tensorflow_probability.python.internal.prefer_static`.
    """
    axis_ = tf.get_static_value(axis)
    if axis_ is None:
        axis_len = tf.shape(x)[axis]
        start = (
            0 if start is None else start
            if start >= 0 else start + axis_len
        )
        stop = (
            axis_len if stop is None else stop
            if stop >= 0 else stop + axis_len
        )
        return tf.gather(x, tf.range(start, stop, delta=step), axis=axis)

    axis = int(axis_)
    if axis >= 0:
        slices = [slice(None)] * axis + [slice(start, stop, step)]
    else:
        slices = [
            Ellipsis, slice(start, stop, step)
        ] + [slice(None)] * (-1 - axis)
    return x[tuple(slices)]


def _static_rank(input):
    """Source:
    `tensorflow_probability.python.internal.prefer_static.rank`.
    """
    if not hasattr(input, "shape"):
        input = (
            tf.convert_to_tensor(input) if tf.get_static_value(input) is None
            else np.array(input)
        )
    ndims_ = tf.TensorShape(getattr(input, "shape", None)).rank
    return tf.rank(input) if ndims_ is None else np.array(ndims_, np.int32)


def _interleave(a, b, axis):
    """Source: is a part of
    `tensorflow_probability.math.scan_associative`.
    """
    num_elems_a = tf.shape(a)[axis]
    num_elems_b = tf.shape(b)[axis]

    axis = np.where(axis >= 0, axis, _static_rank(a)+axis)
    axis = (
        int(axis) if tf.get_static_value(axis) is not None else axis
    )

    def _interleave_with_b(a):
        return tf.reshape(
            tf.concat([
                tf.expand_dims(a, axis=axis+1),
                tf.expand_dims(b, axis=axis+1)
            ], axis=axis+1),
            tf.concat([
                    tf.shape(a)[:axis],
                    [2 * num_elems_b],
                    tf.shape(a)[axis+1:]
            ], axis=0)
        )

    return _static_cond(
        tf.equal(num_elems_a, num_elems_b + 1),
        lambda: tf.concat([
            _interleave_with_b(_slice_along_axis(a, None, -1, axis=axis)),
            _slice_along_axis(a, -1, None, axis=axis)
        ], axis=axis),
        lambda: _interleave_with_b(a),
    )


def scan_associative_lru_optimized(
    lambdas: tf.Tensor,
    elems: tf.Tensor,
    max_num_levels: int = 48,
    axis: int = 0,
) -> tf.Tensor:
    """Perform a scan with the associative binary operation for the
    recurrence of an LRU, in parallel. Mostly lifted from
    tensorflow-probability.

    First compute all needed powers of lambda and then perform the scan
    using these precomputed powers.

    The original associative scan this version is based on is
    `tensorflow_probability.math.scan_associative`.
    """
    T = tf.unstack(tf.shape(elems))[axis]
    num_lambda_powers = tf.cast(tf.math.floor(
        tf.math.log(tf.cast(T, tf.float32)) / tf.math.log(2.0)
    ), tf.int32)
    exponents = 2 ** tf.range(num_lambda_powers, dtype=tf.int32)
    lambda_powers = tf.pow(
        lambdas[None, :],
        tf.cast(exponents[:, None], lambdas.dtype),
    )

    def fn(a, b, tree_level):
        return lambda_powers[tree_level] * a + b

    slice_elem = functools.partial(_slice_along_axis, axis=axis)

    def lowered_fn(a, b, tree_level):
        return tf.nest.flatten(fn(
            tf.nest.pack_sequence_as(elems, a),
            tf.nest.pack_sequence_as(elems, b),
            tree_level=tree_level,
        ))

    def _scan(level, elems):
        elem_length = tf.shape(elems[0])[axis]

        a = [slice_elem(elem, 0, -1, step=2) for elem in elems]
        b = [slice_elem(elem, 1, None, step=2) for elem in elems]
        reduced_elems = lowered_fn(a, b, tree_level=level)

        def handle_base_case_elem_length_two():
            return [
                tf.concat(
                    [slice_elem(elem, 0, 1), reduced_elem],
                    axis=axis,
                ) for (reduced_elem, elem) in zip(reduced_elems, elems)
            ]

        def handle_base_case_elem_length_three():
            reduced_reduced_elems = lowered_fn(
                reduced_elems,
                [slice_elem(elem, 2, 3) for elem in elems],
                tree_level=level,
            )
            return [
                tf.concat([
                    slice_elem(elem, 0, 1),
                    reduced_elem,
                    reduced_reduced_elem
                ], axis=axis) for (reduced_reduced_elem, reduced_elem, elem)
                in zip(reduced_reduced_elems, reduced_elems, elems)
            ]

        at_base_case = tf.logical_or(
            tf.equal(elem_length, 2),
            tf.equal(elem_length, 3),
        )
        base_value = lambda: _static_cond(
            tf.equal(elem_length, 2),
            handle_base_case_elem_length_two,
            handle_base_case_elem_length_three,
        )

        if level >= max_num_levels - 1:
            return base_value()

        def recursive_case():
            odd_elems = _scan(level + 1, reduced_elems)

            def even_length_case():
                return lowered_fn(
                    [slice_elem(odd_elem, 0, -1) for odd_elem in odd_elems],
                    [slice_elem(elem, 2, None, 2) for elem in elems],
                    tree_level=level,
                )

            def odd_length_case():
                return lowered_fn(
                    [odd_elem for odd_elem in odd_elems],
                    [slice_elem(elem, 2, None, 2) for elem in elems],
                    tree_level=level,
                )

            results = _static_cond(
                tf.equal(elem_length % 2, 0),
                even_length_case,
                odd_length_case,
            )
            even_elems = [
                tf.concat([slice_elem(elem, 0, 1), result], axis=axis)
                for (elem, result) in zip(elems, results)
            ]
            return list(map(
                lambda a, b: _interleave(a, b, axis=axis),
                even_elems,
                odd_elems,
            ))

        return _static_cond(at_base_case, base_value, recursive_case)

    return _scan(0, [elems])[0]


def scan_cummax(
    elems: tf.Tensor,
    max_num_levels: int = 48,
    axis: int = 0,
) -> tf.Tensor:
    def fn(a, b):
        return tf.maximum(a, b)

    slice_elem = functools.partial(_slice_along_axis, axis=axis)

    def lowered_fn(a, b):
        return tf.nest.flatten(fn(
            tf.nest.pack_sequence_as(elems, a),
            tf.nest.pack_sequence_as(elems, b),
        ))

    def _scan(level, elems):
        elem_length = tf.shape(elems[0])[axis]

        a = [slice_elem(elem, 0, -1, step=2) for elem in elems]
        b = [slice_elem(elem, 1, None, step=2) for elem in elems]
        reduced_elems = lowered_fn(a, b)

        def handle_base_case_elem_length_two():
            return [
                tf.concat(
                    [slice_elem(elem, 0, 1), reduced_elem],
                    axis=axis,
                ) for (reduced_elem, elem) in zip(reduced_elems, elems)
            ]

        def handle_base_case_elem_length_three():
            reduced_reduced_elems = lowered_fn(
                reduced_elems,
                [slice_elem(elem, 2, 3) for elem in elems],
            )
            return [
                tf.concat([
                    slice_elem(elem, 0, 1),
                    reduced_elem,
                    reduced_reduced_elem
                ], axis=axis) for (reduced_reduced_elem, reduced_elem, elem)
                in zip(reduced_reduced_elems, reduced_elems, elems)
            ]

        at_base_case = tf.logical_or(
            tf.equal(elem_length, 2),
            tf.equal(elem_length, 3),
        )
        base_value = lambda: _static_cond(
            tf.equal(elem_length, 2),
            handle_base_case_elem_length_two,
            handle_base_case_elem_length_three,
        )

        if level >= max_num_levels - 1:
            return base_value()

        def recursive_case():
            odd_elems = _scan(level + 1, reduced_elems)

            def even_length_case():
                return lowered_fn(
                    [slice_elem(odd_elem, 0, -1) for odd_elem in odd_elems],
                    [slice_elem(elem, 2, None, 2) for elem in elems],
                )

            def odd_length_case():
                return lowered_fn(
                    [odd_elem for odd_elem in odd_elems],
                    [slice_elem(elem, 2, None, 2) for elem in elems],
                )

            results = _static_cond(
                tf.equal(elem_length % 2, 0),
                even_length_case,
                odd_length_case,
            )
            even_elems = [
                tf.concat([slice_elem(elem, 0, 1), result], axis=axis)
                for (elem, result) in zip(elems, results)
            ]
            return list(map(
                lambda a, b: _interleave(a, b, axis=axis),
                even_elems,
                odd_elems,
            ))

        return _static_cond(at_base_case, base_value, recursive_case)

    return _scan(0, [elems])[0]
