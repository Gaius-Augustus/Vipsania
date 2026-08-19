from typing import Literal

import tensorflow as tf
from hidten.config import with_config

from .layer import VipsaniaLayer, VipsaniaLayerConfig


class FFNConfig(VipsaniaLayerConfig):

    type: Literal["mlp", "glu"]

    units: int
    d_out: int | None = None
    activation_hidden: str = "relu"
    activation_out: str | None = None
    bias: bool = True


@with_config(FFNConfig)
class MLP(VipsaniaLayer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = FFNConfig(**kwargs)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        d_out = (self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        self.f1 = tf.keras.layers.Dense(
            self.config.units,
            activation=self.config.activation_hidden,
            use_bias=self.config.bias,
        )
        self.f1.build(input_shape)
        self.f2 = tf.keras.layers.Dense(
            d_out,
            activation=self.config.activation_out,
            use_bias=self.config.bias,
        )
        self.f2.build(input_shape[:-1] + (self.config.units, ))
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.preprocess(x, training=training)
        x = self.f2(self.f1(x))
        x = self.postprocess(x, training=training)
        return x

    def _compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        d_out = (
            self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        return input_shape[:-1] + (d_out, )


@with_config(FFNConfig)
class GLUFFN(VipsaniaLayer):
    """Calculates a gated linear unit and an additional dense layer.
    ```
        GLUFFN(x) = W(sigma(Ux) * Vx)
    ```
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = FFNConfig(**kwargs)
        self.W = tf.keras.layers.Dense(
            self.config.units,
            use_bias=self.config.bias,
            activation=self.config.activation_hidden,
        )
        self.V = tf.keras.layers.Dense(
            self.config.units,
            use_bias=self.config.bias,
        )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        d_out = (self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        self.W.build(input_shape)
        self.V.build(input_shape)
        self.Z = tf.keras.layers.Dense(
            d_out,
            use_bias=self.config.bias,
            activation=self.config.activation_out,
        )
        self.Z.build(input_shape[:-1] + (self.config.units, ))
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.preprocess(x, training=training)
        x = self.Z(self.W(x) * self.V(x))
        x = self.postprocess(x, training=training)
        return x

    def _compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        d_out = (
            self.config.d_out if self.config.d_out is not None
            else input_shape[-1]
        )
        return input_shape[:-1] + (d_out, )
