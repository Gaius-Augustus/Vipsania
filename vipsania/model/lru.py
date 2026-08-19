import math
from typing import Any

import tensorflow as tf
from hidten.config import with_config
from pydantic import model_validator

from .layer import VipsaniaLayer, VipsaniaLayerConfig
from .scan import scan_associative_lru_optimized


def associative_scan_op(q_i, q_j):
    """Binary operator used in the parallel scan of the linear
    recurrence.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


class DiskRingRadius(tf.keras.initializers.Initializer):

    def __init__(self, radii: list[tuple[float, float, float]]) -> None:
        self.radii = radii
        self.r_min, self.r_max, self.mixing_ratio = (
            [x[0] for x in radii],
            [x[1] for x in radii],
            [x[2] for x in radii],
        )
        self.mixing_ratio = [
            x / sum(self.mixing_ratio) for x in self.mixing_ratio
        ]

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        dtype = tf.float32 if dtype is None else dtype
        sizes = tf.cast(
            tf.cast(self.mixing_ratio, tf.float32) * shape[0],
            tf.int32,
        )
        sizes = tf.concat(
            [(shape[0] - tf.reduce_sum(sizes[1:]))[tf.newaxis], sizes[1:]],
            axis=0,
        )
        nu_log = tf.constant([], dtype=dtype)
        for i in range(len(self.mixing_ratio)):
            u1 = tf.random.uniform(shape=(sizes[i],), dtype=dtype)
            new_nu_log = tf.math.log(
                -0.5 * tf.math.log(
                    u1 * (self.r_max[i]**2 - self.r_min[i]**2)
                    + self.r_min[i]**2
                )
            )
            nu_log = tf.concat([nu_log, new_nu_log], axis=0)
        return nu_log

    def get_config(self):
        return {'radii': self.radii}


class DiskRingPhase(tf.keras.initializers.Initializer):

    def __init__(self, phases: list[tuple[float, float, float]]) -> None:
        self.phases = phases
        self.min_phase, self.max_phase, self.mixing_ratio = (
            [x[0] for x in phases],
            [x[1] for x in phases],
            [x[2] for x in phases],
        )
        self.mixing_ratio = [
            x / sum(self.mixing_ratio) for x in self.mixing_ratio
        ]

    def __call__(self, shape, dtype=None) -> tf.Tensor:
        dtype = tf.float32 if dtype is None else dtype
        sizes = tf.cast(
            tf.cast(self.mixing_ratio, tf.float32) * shape[0],
            tf.int32,
        )
        if len(self.mixing_ratio) > 1:
            sizes = tf.concat([
                shape[0] - tf.reduce_sum(sizes[1:], keepdims=True),
                sizes[1:]
            ], axis=0)
        theta_log = tf.constant([], dtype=dtype)
        for i in range(len(self.mixing_ratio)):
            u2 = tf.random.uniform(shape=(sizes[i], ), dtype=dtype)
            phase = (
                self.max_phase[i] - self.min_phase[i]
            ) * u2 + self.min_phase[i]
            if self.min_phase[i] < 0:
                phase = tf.where(phase < 0, phase + 6.28, phase)
            theta_log = tf.concat([theta_log, tf.math.log(phase)], axis=0)
        return theta_log

    def get_config(self):
        return {'phases': self.phases}


class LRUConfig(VipsaniaLayerConfig):

    d_latent: int
    d_out: int | None = None

    init_radius: list[tuple[float, float]] = [(0, 1)]
    """Initial min and max radius of the eigenvalues. This can be a list
    of several radii for distribution mixing."""
    init_phase: list[tuple[float, float]] = [(0, math.pi / 10)]
    """Initial min and max phase of the eigenvalues. This can be a list
    of several phases for distribution mixing."""
    init_mixing_ratio: list[float] | None = None
    """If multiple initial radii and phases are given, specifies ratios
    for the number of eigenvalues in each initial region. Defaults to
    uniform distribution over all regions."""

    exp_theta: bool = False
    """Whether to use an exponential transform on the frequency of the
    eigenvalues."""
    initial_state_memory: float | None = None
    """Memorize and adapt an initial state of the LRU recursion for
    improved length generalization. The given float is the rate of
    updating the internal memory."""

    max_tree_depth: int = 14

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_init(self) -> "LRUConfig":
        if len(self.init_radius) != len(self.init_phase):
            raise ValueError(
                "Number of given initial radii and phases are not equal."
            )
        if self.init_mixing_ratio is None:
            n = len(self.init_radius)
            self.init_mixing_ratio = [1/n for _ in range(n)]
        else:
            if len(self.init_mixing_ratio) != len(self.init_radius):
                raise ValueError(
                    "Number of given initial radii and mixing ratios are"
                    " not equal."
                )
        return self


@with_config(LRUConfig)
class LRU(tf.keras.layers.Layer):
    """The Linear Recurrent Unit described in
    "Resurrecting Recurrent Neural Networks for Long Sequences" ~
    Orvieto et. al
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = LRUConfig(**kwargs)
        self.tree_depth = self.config.max_tree_depth

    def set_tree_depth(self, tree_depth: int) -> None:
        self.tree_depth = tree_depth

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        d_out = (self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        stddev = tf.sqrt(2 / (self.config.d_latent + input_shape[-1]))
        self.B_re = self.add_weight(
            shape=(self.config.d_latent, input_shape[-1]),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev/2),
            dtype=tf.float32,
            trainable=True,
            name="B_re",
        )
        self.B_im = self.add_weight(
            shape=(self.config.d_latent, input_shape[-1]),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev/2),
            dtype=tf.float32,
            trainable=True,
            name="B_im",
        )
        stddev = tf.sqrt(2 / (d_out + self.config.d_latent))
        self.C_re = self.add_weight(
            shape=(d_out, self.config.d_latent),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev/2),
            dtype=tf.float32,
            trainable=True,
            name="C_re",
        )
        self.C_im = self.add_weight(
            shape=(d_out, self.config.d_latent),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev/2),
            dtype=tf.float32,
            trainable=True,
            name="C_im",
        )
        self.nu_log = self.add_weight(
            shape=(self.config.d_latent, ),
            initializer=DiskRingRadius([(r[0], r[1], m) for r, m in zip(
                self.config.init_radius, self.config.init_mixing_ratio
            )]),
            trainable=True,
            name="nu_log",
        )
        self.theta_log = self.add_weight(
            shape=(self.config.d_latent, ),
            initializer=DiskRingPhase([(p[0], p[1], m) for p, m in zip(
                self.config.init_phase, self.config.init_mixing_ratio
            )]),
            trainable=True,
            name="theta_log",
        )

        if self.config.initial_state_memory is not None:
            self.q_0_re = self.add_weight(
                shape=(self.config.d_latent, ),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
                dtype=tf.float32,
            )
            self.q_0_im = self.add_weight(
                shape=(self.config.d_latent, ),
                initializer=tf.keras.initializers.Zeros(),
                trainable=False,
                dtype=tf.float32,
            )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        A = tf.exp(tf.complex(
            real=-tf.exp(self.nu_log),
            imag=(tf.exp(self.theta_log) if self.config.exp_theta
                  else self.theta_log),
        ))
        gamma = tf.complex(tf.sqrt(tf.maximum(1 - tf.abs(A)**2, 0.0)), 0.0)
        B_norm = (
            tf.complex(real=self.B_re, imag=self.B_im)
            * tf.expand_dims(gamma, axis=-1)
        )
        Bu = tf.matmul(
            tf.complex(x, 0.),
            B_norm,
            transpose_b=True,
        )
        x = scan_associative_lru_optimized(
            lambdas=A,
            elems=Bu,
            max_num_levels=self.tree_depth,
            axis=-2,
        )
        if training and (m := self.config.initial_state_memory) is not None:
            T = tf.shape(x)[-2]
            A_t = tf.pow(
                tf.tile(A[tf.newaxis, :], [T, 1]),
                tf.cast(tf.range(T)[:, tf.newaxis], tf.complex64),
            )
            Aq_0 = A_t * tf.expand_dims(
                tf.complex(real=self.q_0_re, imag=self.q_0_im), axis=0,
            )
            x = x + Aq_0
            next_q_0 = tf.reduce_mean(x[:, -1, :], axis=0)
            self.q_0_re.assign(
                m * self.q_0_re + (1 - m) * tf.math.real(next_q_0)
            )
            self.q_0_im.assign(
                m * self.q_0_im + (1 - m) * tf.math.imag(next_q_0)
            )
        C = tf.complex(self.C_re, self.C_im)
        x = tf.math.real(tf.matmul(x, C, transpose_b=True))
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        d_out = (self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        return input_shape[:-1] + (d_out, )


class BidirectionalLRUConfig(LRUConfig):

    share_directions: bool = False
    """Share the LRU layer for forward and backward input."""
    reverse_latent: bool = False
    """Whether to also reverse the latent dimension of the reverse
    input."""

    def parent_config(self) -> dict[str, Any]:
        parent_fields = LRUConfig.model_fields.keys()
        return {
            k: v for k, v in self.model_dump().items()
            if k in parent_fields
        }


@with_config(LRUConfig)
class BidirectionalLRU(VipsaniaLayer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = BidirectionalLRUConfig(**kwargs)
        self.lru_fw = LRU(**self.config.parent_config())
        if not self.config.share_directions:
            self.lru_bw = LRU(**self.config.parent_config())

    def set_tree_depth(self, tree_depth: int) -> None:
        self.lru_fw.set_tree_depth(tree_depth)
        if not self.config.share_directions:
            self.lru_bw.set_tree_depth(tree_depth)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.lru_fw.build(input_shape)
        if not self.config.share_directions:
            self.lru_bw.build(input_shape)
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.preprocess(x, training=training)
        y_fw = self.lru_fw(x, training=training)
        x_bw = tf.reverse(x, axis=[-2])
        if self.config.reverse_latent:
            x_bw = tf.reverse(x_bw, axis=[-1])
        if self.config.share_directions:
            y_bw = self.lru_fw(x_bw, training=training)
        else:
            y_bw = self.lru_bw(x_bw, training=training)
        if self.config.reverse_latent:
            y_bw = tf.reverse(y_bw, axis=[-1])
        x = tf.concat([y_fw, tf.reverse(y_bw, axis=[-2])], axis=-1)
        x = self.postprocess(x, training=training)
        return x

    def _compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        d_out = self.lru_fw.compute_output_shape(input_shape)[-1]
        return input_shape[:-1] + (2*d_out, )
