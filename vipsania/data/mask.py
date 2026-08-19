import bricks2marble as b2m
import numpy as np
import tensorflow as tf


def pseudo_mask(x: tf.Tensor) -> tf.Tensor:
    with tf.device("CPU:0"):
        zeros_shape = tf.concat((tf.shape(x)[:-1], [1]), axis=0) # type: ignore
        zeros = tf.zeros(zeros_shape, dtype=x.dtype)  # type: ignore
        return tf.concat((x, zeros), axis=-1)  # type: ignore


def token_masking_sequentially(
    x: tf.Tensor,
    every: int,
    offset: int = 0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Returns a masked version of the input where every ``every``-th
    position is replaced by a masking token, starting at position
    ``offset``.
    """
    B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]  # type: ignore

    indices = tf.range(T)
    mask = tf.math.equal(indices % every - offset, 0)

    mask = tf.tile(tf.expand_dims(mask, axis=0), [B, 1])
    masked_token = tf.concat([
        tf.zeros(D, dtype=x.dtype),
        tf.constant([1], dtype=x.dtype),
    ], 0)
    return mask, tf.where(
        tf.expand_dims(mask, -1),
        masked_token,
        tf.concat([x, tf.zeros((B, T, 1), dtype=x.dtype)], -1),
    )


def token_masking(
    x: tf.Tensor,
    y: tf.Tensor,
    masking: float,
    token: float | None = None,
    same: float | None = None,
    false: float | None = None,
    ignore_y_values: tuple[int, ...] = (-1, ),
    seed: int | None = None,
    weight_repeat_masked: float = 0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor]:
    """Masks a given input and expected output tensor. This increases
    the last dimension of the input by 1, which is used as the masking
    indicator.
    """
    with tf.device("CPU:0"):
        if masking <= 0:
            return pseudo_mask(x), y, tf.ones_like(y)
        expand_flag = False
        if x.ndim is not None and y.ndim is not None and (
            x.ndim == y.ndim+1 == 2
        ):
            expand_flag = True
            x = tf.expand_dims(x, axis=0)
            y = tf.expand_dims(y, axis=0)
        dist = [token, same, false]
        assign = dist.count(None)
        specified_sum = sum([v for v in dist if v is not None])
        if specified_sum > 1:
            raise ValueError("Specified probabilities sum to more than 1.")
        if assign == 0 and specified_sum < 1:
            raise ValueError("Specified probabilities sum to less than 1.")
        fill_value = (1 - specified_sum) / assign if assign > 0 else 0
        dist = [v if v is not None else fill_value for v in dist]
        p_mask, p_same, _ = dist

        B, T, D = tf.unstack(tf.shape(x))  # type: ignore

        n_masked = tf.cast(tf.cast(T, tf.float32) * masking, tf.int32)
        # Shuffling a mask is done multiple times in this function and there
        # are two ways to do that: Either with tf.map_fn(shuffle, tensor) or
        # the approach below. The second approach has the disadvantage that
        # each row gets permuted in the same way, which might lead to
        # unwanted training behaviour. This is something we have to look out
        # for, maybe change to the first approach later on.
        mask = tf.transpose(tf.random.shuffle(
            tf.concat([
                tf.ones((n_masked, B), dtype=tf.bool),
                tf.zeros((T - n_masked, B), dtype=tf.bool)
            ], axis=0),
            seed=seed,
        ))

        n_mask_mask = tf.cast(
            tf.math.ceil(tf.cast(n_masked, tf.float32) * p_mask),
            tf.int32,
        )
        mask_indices = tf.where(mask)
        mask_mask_flat = tf.reshape(tf.transpose(tf.random.shuffle(
            tf.concat([
                tf.ones((n_mask_mask, B), dtype=tf.bool),
                tf.zeros((n_masked - n_mask_mask, B), dtype=tf.bool)
            ], axis=0),
            seed=seed,
        )), [-1])
        mask_mask = tf.zeros_like(mask, dtype=tf.bool)
        mask_mask = tf.tensor_scatter_nd_update(
            mask_mask,
            mask_indices,
            mask_mask_flat,
        )

        if p_same > 0:
            mask_same = tf.logical_and(mask, tf.logical_not(mask_mask))
            n_mask_same = tf.cast(
                tf.math.floor(tf.cast(n_masked, tf.float32) * p_same),
                tf.int32,
            )
            def get_mask_same_row(row):
                n = tf.reduce_sum(tf.cast(row, tf.int32))
                same_mask = tf.concat([
                    tf.ones((n_mask_same,), dtype=tf.bool),
                    tf.zeros((n - n_mask_same,), dtype=tf.bool)
                ], axis=0)
                same_mask = tf.random.shuffle(same_mask, seed=seed)
                return tf.tensor_scatter_nd_update(
                    row,
                    tf.where(row),
                    same_mask,
                )
            mask_same = tf.map_fn(get_mask_same_row, mask_same)
        else:
            mask_same = tf.zeros_like(mask, dtype=tf.bool)

        mask_false = tf.logical_and(mask, tf.logical_not(mask_mask))
        mask_false = tf.logical_and(mask_false, tf.logical_not(mask_same))

        for value in ignore_y_values:
            y_not_N = tf.not_equal(y, value)
            mask_mask = tf.logical_and(mask_mask, y_not_N)
            mask_same = tf.logical_and(mask_same, y_not_N)
            mask_false = tf.logical_and(mask_false, y_not_N)

        not_repeat_masked = tf.case([
            (tf.equal(D, 6), lambda: tf.not_equal(x[:, :, -1], 1)),
            (tf.equal(D, 9), lambda: tf.argmax(x, axis=-1) < 4),
        ], default=lambda: tf.fill(tf.shape(x)[:2], True))
        if weight_repeat_masked > 0:
            sample_weights = tf.where(
                not_repeat_masked,
                1.0,
                weight_repeat_masked,
            )
        else:
            sample_weights = tf.ones_like(y, dtype=tf.float32)
            mask_mask = tf.logical_and(mask_mask, not_repeat_masked)
            mask_same = tf.logical_and(mask_same, not_repeat_masked)
            mask_false = tf.logical_and(mask_false, not_repeat_masked)

        mask_mask_exp = tf.expand_dims(mask_mask, axis=-1)
        x_masked = tf.where(
            tf.broadcast_to(mask_mask_exp, tf.shape(x)),
            tf.cast(0.0, x.dtype),
            x,
        )
        x = tf.concat([x_masked, tf.cast(mask_mask_exp, x.dtype)], axis=-1)

        flat_mask_false = tf.reshape(mask_false, [-1])
        n_false = tf.reduce_sum(tf.cast(flat_mask_false, tf.int32))
        rand_indices = tf.random.uniform(
            (n_false, ),
            minval=0,
            maxval=D,
            dtype=tf.int32,
            seed=seed,
        )
        encoding = tf.case([
            (tf.equal(D, 6), lambda: np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 1],
            ])),
        ], default=lambda: tf.eye(D, dtype=tf.int64))
        false_tokens = tf.gather(
            encoding,
            rand_indices,
        )
        false_tokens = tf.cast(tf.concat([
            false_tokens,
            tf.zeros((n_false, 1), dtype=false_tokens.dtype)
        ], axis=-1), x.dtype)

        x = tf.reshape(
            tf.tensor_scatter_nd_update(
                tf.reshape(x, [-1, tf.shape(x)[-1]]),
                tf.where(flat_mask_false),
                false_tokens,
            ),
            tf.shape(x),
        )

        y = tf.where(
            mask_mask | mask_same | mask_false,
            y,
            -1,
        )
        if expand_flag:
            x = x[0]
            y = y[0]
            sample_weights = sample_weights[0]
        return x, y, sample_weights


def mask_positions(arr: np.ndarray, pos: np.ndarray):
    """Masks the positions of the given one-hot encoded nucleotide
    sequence by adding a row to this array at the given positions.

    Args:
        arr (np.ndarray): Array with one-hot encoded nucleotides of
            shape ``(T, D)``.
        pos (np.ndarray): Array of shape ``(S, )`` with indices of
            positions in ``arr`` that will be masked.

    Returns:
        np.ndarray: Masked array of shape ``(T, D+1)``.
    """
    mask = np.zeros(arr.shape[0])
    mask[pos] = 1
    return np.concatenate((arr, np.expand_dims(mask, -1)), axis=-1)


def token_masking_np(
    self,
    x: np.ndarray,
    y: np.ndarray,
    p: float = .15,
    p_mask: float = .8,
    p_same: float = .1,
    p_false: float = .1,
) -> tuple[np.ndarray, np.ndarray]:
    B, T, _ = x.shape
    mask = np.full((B, T), False)
    n_masked = int(T*p)
    mask[:, :n_masked] = True
    if self._seed is not None:
        rng = np.random.default_rng(seed=self._seed)
    else:
        rng = np.random.default_rng()
    mask = rng.permuted(mask, axis=1)

    mask_mask = mask.copy()
    mask_mask_help = np.full((B, n_masked), False)
    mask_mask_help[:, :int(n_masked*p_mask)] = True
    mask_mask_help = rng.permuted(mask_mask_help, axis=1)
    mask_mask[mask] = mask_mask_help.reshape(-1)
    mask_same = mask & ~mask_mask
    mask_b_help = np.full((B, mask_same[0, :].sum()), False)
    mask_b_help[:, :int(n_masked*p_same)] = True
    mask_b_help = rng.permuted(mask_b_help, axis=1)
    mask_same[mask_same] = mask_b_help.reshape(-1)
    mask_false = mask & (~mask_mask) & (~mask_same)

    y_not_N = y != 4
    mask_mask &= y_not_N
    mask_same &= y_not_N
    mask_false &= y_not_N
    if self.repeat_masking:
        mask_mask = mask_mask*np.logical_not(x[:, :, x.shape[2]-1] == 1)
        x[mask_mask, 5] = 0
        mask_same = mask_same*np.logical_not(x[:, :, x.shape[2]-1] == 1)
        x[mask_same, 5] = 0
        mask_false = mask_false*np.logical_not(x[:, :, x.shape[2]-1] == 1)
        x[mask_false, 5] = 0
    mask = mask_mask | mask_same | mask_false

    if self.mask_type == "M":
        x = np.concatenate((x, np.zeros((B, T, 1))), axis=-1)
        x[mask_mask, :-1] = 0
        x[mask_mask, -1] = 1
        x[mask_false, :] = np.concatenate((
            self.encoding[np.random.randint(
                    0,
                    5 if not self.repeat_masking else 9,
                    mask_false.sum(),
            )],
            np.zeros((mask_false.sum(), 1)),
        ), axis=-1)

    y_ingore_index = np.full_like(y, -1)
    y_ingore_index[mask] = y[mask]
    y = y_ingore_index
    return x, y
