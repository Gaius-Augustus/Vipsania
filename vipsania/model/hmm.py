from typing import Any, Literal

import tensorflow as tf
from bricks2marble.tf import AnnotationHMM, AnnotationHMMConfig
from hidten import HMMMode
from hidten.config import ModelConfig, with_config

from .layer import VipsaniaLayer, VipsaniaLayerConfig


def extract_nucleotides(
    x: tf.Tensor,
    uniform_N: bool = False,
    with_repeats: bool = False,
    remove_repeats: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Given a tensor of encoded nucleotides, only extracts the first 5
    (one-hot A,C,G,T,N) characters. Tokens that are set to a token mask
    will also be translated to N.
    """
    if with_repeats:
        nuc = tf.where(
            tf.expand_dims(tf.equal(x[..., -1], 1), axis=-1),  # type: ignore
            tf.constant(
                [1/4, 1/4, 1/4, 1/4, 0]
                    if uniform_N else [0, 0, 0, 0, 1, 0],
                dtype=x.dtype,
            ),
            x[..., :(5 if uniform_N else 6)],  # type: ignore
        )
    else:
        nuc = tf.where(
            tf.expand_dims(tf.equal(x[..., -1], 1), axis=-1),  # type: ignore
            tf.constant(
                [1/4, 1/4, 1/4, 1/4]
                    if uniform_N else [0, 0, 0, 0, 1],
                dtype=x.dtype,
            ),
            x[..., :(4 if uniform_N else 5)],  # type: ignore
        )
    if remove_repeats:
        x = tf.concat([
            x[..., :(4 if uniform_N else 5)],
            x[..., (5 if uniform_N else 6):]
        ], axis=-1)  # type: ignore
    return x, nuc


class HMMBlockConfig(VipsaniaLayerConfig, AnnotationHMMConfig):

    return_posterior: bool = False
    prior_loop: int = 1
    """Number of loops to submit the posterior of the HMM as a prior to
    another HMM call before returning it.
    """

    posterior_log_lower: float | None = None
    parallel_factor: int = 1

    def parent_config(self) -> dict[str, Any]:
        parent_fields = AnnotationHMMConfig.model_fields.keys()
        return {
            k: v for k, v in self.model_dump().items()
            if k in parent_fields
        }


@with_config(HMMBlockConfig)
class HMMBlock(VipsaniaLayer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = HMMBlockConfig(**kwargs)
        self._infer = False
        self._infer_mode = HMMMode.VITERBI

        self.parallel = self.config.parallel_factor

    def set_parallel(self, parallel: int) -> None:
        self.parallel = parallel

    def toggle_inference(
        self,
        inference: bool | None = None,
        mode: HMMMode | Literal["A"] = HMMMode.VITERBI,
    ) -> bool:
        if inference is None: self._infer = not self._infer
        else: self._infer = inference
        self._infer_mode = mode
        return self._infer

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        hmmlayer_config = self.config.parent_config()
        if hmmlayer_config.get("transition_scorer", None) is not None:
            hmmlayer_config["transition_scorer"] = (
                hmmlayer_config["transition_scorer"] | {
                    "input_size": input_shape[-1]
                }
            )
        self.hmmlayer = AnnotationHMM(**hmmlayer_config)
        self.hmmlayer.build(input_shape[:-1] + (
            self.config.embed if self.config.embed is not None
            else input_shape[-1],
        ))
        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        x_preprocessed = self.preprocess(x, training=training)
        posterior = self.hmmlayer(
            x_preprocessed, nuc,
            mode=(self._infer_mode
                if self._infer and self.config.emitter_prior is None
                else HMMMode.POSTERIOR),
            parallel=self.parallel,
            training=training,
            transition_scores=x,
            prior=tf.ones(tf.concat((
                (tf.shape(x)[0]*(2 if self.config.use_reverse_strand else 1),),
                tf.shape(x)[1:2],
                (self.config.n_states*self.config.heads, )
            ), axis=0)) / self.config.n_states
            if self.config.emitter_prior is not None else None,
        )
        if self._infer and (
            self.config.emitter_prior is None or self.config.prior_loop == 0
        ):
            return posterior
        if self.config.emitter_prior is not None:
            for k in range(self.config.prior_loop):
                if self.config.use_reverse_strand:
                    # B, T, 2*H, D
                    post_resh = tf.reshape(posterior, tf.concat((
                        tf.shape(posterior)[:2],
                        (2, tf.shape(posterior)[2]//2, tf.shape(posterior)[3])
                    ), 0))
                    # B, T, 2, H, D
                    post_resh = tf.keras.ops.moveaxis(post_resh, 2, 0)
                    # 2, B, T, H, D
                    post_resh = tf.concat(
                        (post_resh[:1], tf.reverse(post_resh[1:2], [2])),
                        axis=0,
                    )
                    post_resh = tf.reshape(post_resh, tf.concat(
                        ((2*tf.shape(post_resh)[1],), tf.shape(post_resh)[2:]),
                        axis=0,
                    ))
                    # 2*B, T, H, D
                else:
                    post_resh = posterior
                post_resh = tf.reshape(post_resh, tf.concat((
                    tf.shape(post_resh)[:2],
                    (tf.shape(post_resh)[2] * tf.shape(post_resh)[3], )
                ), 0))
                # (2*)B, T, H*D
                posterior = self.hmmlayer(
                    x_preprocessed, nuc,
                    mode=(self._infer_mode
                        if self._infer and k == self.config.prior_loop - 1
                        else HMMMode.POSTERIOR),
                    parallel=self.parallel,
                    training=training,
                    transition_scores=x,
                    prior=post_resh,
                )
                if self._infer and k == self.config.prior_loop - 1:
                    return posterior

        x = tf.reshape(posterior, tf.concat((
            tf.shape(posterior)[:2],
            (tf.shape(posterior)[2] * tf.shape(posterior)[3], )
        ), 0))
        if self.config.posterior_log_lower is not None:
            x = tf.math.log(tf.maximum(x, self.config.posterior_log_lower))
        x = self.postprocess(x, training=training)
        if training and self.config.return_posterior:
            return x, posterior
        return x

    def _compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        layer_out = self.hmmlayer.compute_output_shape(
            input_shape[:-1] + (
                self.config.embed if self.config.embed is not None
                else input_shape[-1],
        ))
        return layer_out[:-2] + (layer_out[-2] * layer_out[-1], )
