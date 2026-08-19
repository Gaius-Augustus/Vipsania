from typing import Literal

import tensorflow as tf
from hidten.config import ModelConfig, with_config
from pydantic import model_validator


def sample_conv1d(x: tf.Tensor, kernels: tf.Tensor) -> tf.Tensor:
    """Convolution per-sample of a tensor of shape ``(B, T)`` with a
    kernel of shape ``(B, K)``, producing a tensor of shape ``(B, T)``.
    """
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    K = tf.shape(kernels)[1]

    x4d = tf.reshape(tf.transpose(x), [1, T, 1, B])
    k4d = tf.reshape(tf.transpose(kernels), [K, 1, B, 1])

    out = tf.nn.depthwise_conv2d(
        x4d, k4d,
        strides=[1, 1, 1, 1],
        padding="SAME",
    )
    out = tf.transpose(out[0, :, 0, :])
    return out


class SplicedCrossentropyAdapterConfig(ModelConfig):

    source: list[Literal["site", "exon", "gene", "outer", "inner"]]
    weight: list[float]
    kernel: list[tuple[int, float] | None] | None = None
    cutoff: list[float | None] | None = None
    switch: list[bool] | None = None
    target: list[Literal["position", "sequence"]] | None = None
    double: list[bool] | None = None
    adaptk: list[tuple[float, int] | None] | None = None

    memory: float = 0.99

    @model_validator(mode="after")
    def populate_lists(self):
        if self.kernel is None:
            self.kernel = [None for _ in self.source]
        if self.cutoff is None:
            self.cutoff = [None for _ in self.source]
        if self.switch is None:
            self.switch = [False for _ in self.source]
        if self.target is None:
            self.target = ["position" for _ in self.source]
        if self.double is None:
            self.double = [True for _ in self.source]
        if self.adaptk is None:
            self.adaptk = [None for _ in self.source]
        return self


@with_config(SplicedCrossentropyAdapterConfig)
class SplicedCrossentropyAdapter(tf.keras.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = SplicedCrossentropyAdapterConfig(**kwargs)

        self.loss_weights = []
        for w in self.config.weight:
            self.loss_weights.append(tf.Variable(
                w,
                trainable=False,
                dtype=tf.float32,
            ))

        self.conv_kernel_width = []
        self.conv_kernel_std = []
        for i in range(len(self.config.source)):
            if (
                self.config.kernel[i] is not None
                and self.config.adaptk[i] is None
            ):
                self.conv_kernel_width.append(tf.Variable(
                    self.config.kernel[i][0],
                    trainable=False,
                    dtype=tf.float32,
                ))
                self.conv_kernel_std.append(tf.Variable(
                    self.config.kernel[i][1],
                    trainable=False,
                    dtype=tf.float32,
                ))
            else:
                self.conv_kernel_width.append(None)
                self.conv_kernel_std.append(None)

        self.memory = []
        for _ in self.config.source:
            m = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.memory.append(m)

    def call(self, p: tf.Tensor, sample_weight: tf.Tensor) -> tf.Tensor:
        if "site" in self.config.source:
            P_site = tf.reduce_mean(tf.reduce_sum(p[..., -8:], -1), -1)
        if "exon" in self.config.source:
            P_exon = tf.reduce_mean(tf.reduce_sum(p[..., -11:], -1), -1)
        if "gene" in self.config.source:
            P_gene = tf.reduce_mean(tf.reduce_sum(p[..., 1:], -1), -1)
        if "outer" in self.config.source:
            P_outer = tf.reduce_mean(p[..., -8] + p[..., -1], -1)
        if "inner" in self.config.source:
            P_inner = tf.reduce_mean(tf.reduce_sum(p[..., -7:-1], -1), -1)

        factor = 1 - tf.reduce_sum(self.loss_weights)
        for i, s in enumerate(self.config.source):
            match s:
                case "site": prob = P_site
                case "exon": prob = P_exon
                case "gene": prob = P_gene
                case "outer": prob = P_outer
                case "inner": prob = P_inner

            if self.config.double[i]:
                prob = 2*prob

            if self.config.cutoff[i] is not None and self.config.switch[i]:
                prob = tf.cast(prob > self.config.cutoff[i], tf.float32)

            if self.conv_kernel_width[i] is not None:
                ranges = tf.range(
                    -tf.cast(self.conv_kernel_width[i], tf.int32) // 2 + 1,
                    tf.cast(self.conv_kernel_width[i], tf.int32) // 2 + 1,
                    dtype=tf.float32,
                )
                kernel = tf.exp(-0.5 * (ranges / self.conv_kernel_std[i])**2)
                prob = tf.nn.conv1d(
                    tf.expand_dims(prob, -1),
                    tf.expand_dims(tf.expand_dims(kernel, -1), -1),
                    stride=1,
                    padding="SAME",
                )[:, :, 0]
                if self.config.cutoff[i] is not None and self.config.switch[i]:
                    prob = tf.cast(prob > 1e-7, tf.float32)
            elif self.config.adaptk[i] is not None:
                c = self.config.adaptk[i][1] // 2
                w = self.config.adaptk[i][0] / (
                    tf.reduce_mean(prob, axis=-1) + 1e-5
                )
                w = tf.cast(tf.clip_by_value(w / 2, 1, c), tf.int32)
                positions = tf.range(self.config.adaptk[i][1], dtype=tf.int32)
                dist_from_centre = tf.broadcast_to(
                    tf.expand_dims(tf.abs(positions - c), 0),
                    tf.concat((tf.shape(w), tf.shape(positions)), axis=0),
                )
                kernel = tf.cast(
                    dist_from_centre <= tf.expand_dims(w, -1),
                    tf.float32,
                )
                prob = sample_conv1d(prob, kernel)
                if self.config.cutoff[i] is not None and self.config.switch[i]:
                    prob = tf.cast(prob > 1e-7, tf.float32)

            if self.config.cutoff[i] is not None and not self.config.switch[i]:
                prob = tf.cast(prob > self.config.cutoff[i], tf.float32)

            if self.config.target[i] == "sequence":
                prob = tf.reduce_max(prob, axis=1, keepdims=True)

            self.memory[i].assign(
                self.config.memory * self.memory[i]
                + (1 - self.config.memory) * tf.reduce_mean(prob)
            )
            factor = factor + self.loss_weights[i] * prob

        return sample_weight * factor
