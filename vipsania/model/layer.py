from typing import Literal

import tensorflow as tf
from hidten.config import ModelConfig, with_config

from ..hooks import with_hooks


class Derf(tf.keras.Layer):

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.gamma = self.add_weight((input_shape[-1], ), initializer="ones")
        self.beta = self.add_weight((input_shape[-1], ), initializer="zeros")
        self.alpha = self.add_weight(
            initializer=tf.keras.initializers.Constant(0.5),
        )
        self.s = self.add_weight(
            initializer=tf.keras.initializers.Constant(0),
        )

    def call(self, x):
        return self.gamma * tf.math.erf(self.alpha * x + self.s) + self.beta


class VipsaniaLayerConfig(ModelConfig):

    embed: int | None = None
    embed_norm: Literal["layer", "batch", "RMS", "Derf"] | None = None
    embed_bias: bool = False
    embed_activation: str | None = None

    dropout: float = 0

    readout: int | bool = False
    readout_type: Literal["dense", "conv"] = "dense"
    readout_conv_kernel: int = 9
    readout_norm: Literal["layer", "batch", "RMS", "Derf"] | None = None
    readout_bias: bool = False
    pre_readout_activation: str | None = None
    readout_activation: str | None = None


@with_hooks
@with_config(VipsaniaLayerConfig)
class VipsaniaLayer(tf.keras.Layer):

    def build(self, input_shape: tuple[int | None,  ...]) -> None:
        output_shape = self._compute_output_shape(input_shape)

        if self.config.embed_norm == "layer":
            self.norm = tf.keras.layers.LayerNormalization()
        elif self.config.embed_norm == "batch":
            self.norm = tf.keras.layers.BatchNormalization()
        elif self.config.embed_norm == "RMS":
            self.norm = tf.keras.layers.LayerNormalization(rms_scaling=True)
        elif self.config.embed_norm == "Derf":
            self.norm = Derf()

        if self.config.embed_norm is not None:
            self.norm.build(input_shape)

        if self.config.embed is not None:
            self.embedding = tf.keras.layers.Dense(
                self.config.embed,
                use_bias=self.config.embed_bias,
                activation=self.config.embed_activation,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
            )
            self.embedding.build(input_shape)

        if self.config.dropout > 0:
            self.dropout = tf.keras.layers.Dropout(self.config.dropout)

        if self.config.readout_norm == "layer":
            self.readout_norm = tf.keras.layers.LayerNormalization()
        elif self.config.readout_norm == "batch":
            self.readout_norm = tf.keras.layers.BatchNormalization()
        elif self.config.readout_norm == "RMS":
            self.readout_norm = tf.keras.layers.LayerNormalization(
                rms_scaling=True,
            )
        elif self.config.readout_norm == "Derf":
            self.readout_norm = Derf()

        if self.config.readout_norm is not None:
            self.readout_norm.build(output_shape)

        if self.config.pre_readout_activation is not None:
            self.pre_readout_activation = tf.keras.activations.get(
                self.config.pre_readout_activation,
            )

        if self.config.readout is not False:
            d_out = (
                input_shape[-1] if self.config.readout is True
                else self.config.readout
            )
            if self.config.readout_type == "dense":
                self.readout = tf.keras.layers.Dense(
                    d_out,
                    use_bias=self.config.readout_bias,
                    activation=self.config.readout_activation,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                )
            elif self.config.readout_type == "conv":
                self.readout = tf.keras.layers.Conv1D(
                    d_out,
                    self.config.readout_conv_kernel,
                    padding="same",
                    use_bias=self.config.readout_bias,
                    activation=self.config.readout_activation,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                )
            self.readout.build(output_shape)

    def preprocess(self, x, training: bool = False):
        if self.config.embed_norm is not None:
            x = self.norm(x)
        if self.config.embed is not None:
            x = self.embedding(x)
        return x

    def postprocess(self, x, training: bool = False):
        if self.config.dropout > 0:
            x = self.dropout(x, training=training)
        if self.config.readout_norm is not None:
            x = self.readout_norm(x)
        if self.config.pre_readout_activation is not None:
            x = self.pre_readout_activation(x)
        if self.config.readout is not False:
            x = self.readout(x)
        return x

    def _compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        if self.config.readout is True: return input_shape
        shape = self._compute_output_shape(input_shape)
        if self.config.readout is False: return shape
        return shape[:-1] + (self.config.readout, )
