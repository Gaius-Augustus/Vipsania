from typing import Generator, Literal

import tensorflow as tf
from hidten import HMMMode
from hidten.config import ModelConfig, with_config

from ..hooks import with_hooks
from .ffn import GLUFFN, MLP, FFNConfig
from .hmm import HMMBlock, HMMBlockConfig, extract_nucleotides
from .layer import Derf
from .loss import SplicedCrossentropyAdapter, SplicedCrossentropyAdapterConfig
from .lru import BidirectionalLRU, BidirectionalLRUConfig


class VipsaniaStripeConfig(ModelConfig):

    order: list[str]
    residual: bool = True

    lru: BidirectionalLRUConfig | None = None
    hmm: HMMBlockConfig | None = None
    ffn: FFNConfig | None = None

    return_posterior: bool = False

    model_config = {"frozen": True, "extra": "forbid"}


@with_config(VipsaniaStripeConfig)
class VipsaniaStripe(tf.keras.Layer):
    """A type of layer with matching input and output shape. Is normally
    connected by a residual stream. Can be used to group a seq2seq layer
    with an MLP.
    """

    def __init__(self, layer_id: int, stripe_id: int, **kwargs) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.stripe_id = stripe_id
        self.config = VipsaniaStripeConfig(**kwargs)

        if self.config.lru is not None:
            self.lru = BidirectionalLRU(**self.config.lru.model_dump())
        if self.config.hmm is not None:
            self.hmm = HMMBlock(**self.config.hmm.model_dump() | {
                "return_posterior": self.config.return_posterior,
            })
        if self.config.ffn is not None:
            match self.config.ffn.type:
                case "mlp":
                    self.mlp = MLP(**self.config.ffn.model_dump())
                case "glu":
                    self.mlp = GLUFFN(**self.config.ffn.model_dump())

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        shape = input_shape
        for strp in self.config.order:
            if strp == "ffn": strp = "mlp"
            getattr(self, strp).build(shape)
            shape = getattr(self, strp).compute_output_shape(shape)

    def toggle_inference(
        self,
        enable: bool | None = None,
        mode: HMMMode | Literal["A"] = HMMMode.VITERBI,
    ) -> bool:
        if self.config.hmm is None: raise RuntimeError(
            "Attempted to set inference mode for a stripe without an HMM."
        )
        return self.hmm.toggle_inference(enable, mode=mode)

    def call(self, x, nuc, training: bool = False):
        posterior = None
        y = x

        for strp in self.config.order:
            if strp == "ffn": strp = "mlp"
            if strp == "hmm":
                if self.hmm.config.return_posterior and training:
                    y, posterior = self.hmm(y, nuc, training=training)
                else:
                    y = self.hmm(y, nuc, training=training)
                if self.hmm._infer: return y, None
            else:
                y = getattr(self, strp)(y, training=training)

        x = x + y if self.config.residual else y
        return x, posterior

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape


class VipsaniaConfig(ModelConfig):

    d_hidden: int
    n_layers: int
    stripes: list[list[str]]
    restrict_stripes: dict[int, list[int]] = {}
    """Restrict a certain block type to appear only in a selected list
    of layer indices."""

    lru: BidirectionalLRUConfig | None = None
    hmm: HMMBlockConfig | None = None
    ffn: FFNConfig | None = None

    input_repeat_masked: bool = True
    predict_repeat_masked: bool = False

    unembedding_norm: Literal["layer", "batch", "RMS", "Derf"] = "layer"

    spliced_loss: SplicedCrossentropyAdapterConfig | None = None

    model_config = {"frozen": True, "extra": "forbid"}


@with_hooks
@with_config(VipsaniaConfig)
class Vipsania(tf.keras.Model):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = VipsaniaConfig(**kwargs)

        self.embedding = tf.keras.layers.Dense(
            self.config.d_hidden,
            use_bias=False,
        )

        self.stripes: list[list[VipsaniaStripe]] = []
        for lid in range(self.config.n_layers):
            stripes = []
            real_sid = 0
            for sid, strp in enumerate(self.config.stripes):
                if (sid in self.config.restrict_stripes
                    and lid not in self.config.restrict_stripes[sid]
                ): continue
                stripes.append(VipsaniaStripe(lid, real_sid,
                    **{s: getattr(self.config, s).model_dump() for s in strp},
                    order=strp,
                    return_posterior=self.config.spliced_loss is not None,
                ))
                real_sid += 1
            self.stripes.append(stripes)
            if len(self.stripes[-1]) == 0:
                raise RuntimeError(f"No stripes in layer {lid}.")

        if self.config.unembedding_norm == "layer":
            self.unembedding_norm = tf.keras.layers.LayerNormalization()
        elif self.config.unembedding_norm == "batch":
            self.unembedding_norm = tf.keras.layers.BatchNormalization()
        elif self.config.unembedding_norm == "RMS":
            self.unembedding_norm = tf.keras.layers.LayerNormalization(
                rms_scaling=True,
            )
        elif self.config.unembedding_norm == "Derf":
            self.unembedding_norm = Derf()
        self.unembedding = tf.keras.layers.Dense(
            8 if self.config.predict_repeat_masked else 4,
            use_bias=False,
        )

        self._infer = False

    def set_options(
        self,
        parallel: int | None = None,
        tree_depth: int | None = None,
    ) -> None:
        if parallel is not None:
            for stripe in self.get_stripes("hmm"):
                stripe.hmm.set_parallel(parallel)
        if tree_depth is not None:
            for stripe in self.get_stripes("lru"):
                stripe.lru.set_tree_depth(tree_depth)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        BT = input_shape[:-1]
        D_in = input_shape[-1]
        if not self.config.input_repeat_masked:
            D_in -= 1  # type: ignore

        self.embedding.build(BT + (D_in, ))

        for layer in self.stripes:
            for stripe in layer:
                stripe.build(BT + (self.config.d_hidden, ))

        d_unembed = self.config.d_hidden
        self.unembedding_norm.build(BT + (d_unembed, ))
        self.unembedding.build(BT + (d_unembed, ))

        if self.config.spliced_loss is not None:
            self.spliced_loss_adapter = SplicedCrossentropyAdapter(
                **self.config.spliced_loss.model_dump()
            )

    def toggle_inference(
        self,
        mode: HMMMode | Literal["A", "embedding"] = HMMMode.VITERBI,
        layer: int = -1,
        stripe: int = -1,
        hmm: int = -1,
        enable: bool | None = None,
    ) -> None:
        """Configure early return of a specific HMM output or the
        residual stream at given layer and stripe.

        Args:
            mode (HMMMode or 'A'/'embedding', optional): The mode used
                for inference. Can be either a mode for the HMM (see
                :class:`hidten.HMMMode`), the transition matrix 'A' or
                an 'embedding' of the Vipsania model. Defaults to the
                Viterbi algorithm.
            layer (int, optional): Index of the layer to infer the
                embedding from, starting at 0. Defaults to the last
                layer.
            stripe (int, optional): Index of the stripe within the
                targeted layer to infer the embedding from, starting at
                0. Defaults to the last stripe in the layer.
            hmm (int, optional): Index of the HMM to infer. Defaults to
                the last HMM in the model.
            enable (bool, optional): Set this to a boolean to enable or
                disable the embedding directly. Defaults to always
                changing the current mode.
        """
        enable = (not self._infer) if enable is None else enable
        self._infer = enable
        for strp in self.get_stripes("hmm"): strp.toggle_inference(False)
        if not enable: return

        if mode != "embedding":
            strps = list(self.get_stripes("hmm"))
            try:
                strp = strps[hmm]
                layer = strp.layer_id
                stripe = strp.stripe_id
            except IndexError as e:
                raise ValueError(
                    f"HMM index {hmm} out of bounds for {len(strps)} HMM in "
                    "the model."
                ) from e
            strp.toggle_inference(enable, mode=mode)
        else:
            try:
                strp = self.stripes[layer][stripe]
                layer = strp.layer_id
                stripe = strp.stripe_id
            except IndexError as e:
                ids = sorted({
                    (lid, sid) for lid in range(len(self.stripes))
                    for sid in range(len(self.stripes[lid]))
                })
                raise IndexError(
                    f"Layer and stripe index ({layer}, {stripe}) out of "
                    f"bounds. Valid pairs are: {ids}"
                ) from e
        self._infer_layer = layer
        self._infer_stripe = stripe

    def _requires_repeats_in_nuc(self) -> bool:
        flag = False
        for strp in self.get_stripes("hmm"):
            if strp.config.hmm.repeats_in_nuc: flag = True
        return flag

    def _is_uniform_N_in_nuc(self) -> bool:
        flag = False
        for strp in self.get_stripes("hmm"):
            if strp.config.hmm.uniform_N: flag = True
        return flag

    def get_stripes(
        self,
        key: str | None = None,
    ) -> Generator[VipsaniaStripe, None, None]:
        for block in self:
            if key is None or (
                hasattr(block.config, key)
                and getattr(block.config, key) is not None
            ): yield block

    def __iter__(self) -> "Vipsania":
        self._counter = -1
        return self

    def __next__(self) -> VipsaniaStripe:
        self._counter += 1
        i = 0
        for layer in self.stripes:
            for block in layer:
                if self._counter == i: return block
                i += 1
        raise StopIteration

    def call(self, x, training: bool = False):
        posterior = []
        x, nuc = extract_nucleotides(
            x,
            uniform_N=self._is_uniform_N_in_nuc(),
            with_repeats=self._requires_repeats_in_nuc(),
            remove_repeats=not self.config.input_repeat_masked,
        )
        x = self.embedding(x)  # type: ignore
        self.hook("embedding", x)

        for lid, layer in enumerate(self.stripes):
            for sid, stripe in enumerate(layer):
                x, p = stripe(x, nuc, training=training)
                self.hook(f"layer_{lid}_stripe_{sid}", x)
                if p is not None:
                    posterior.append(p)
                if (self._infer
                    and self._infer_layer == lid
                    and self._infer_stripe == sid
                ): return x
            self.hook(f"layer_{lid}", x)

        x = self.unembedding_norm(x)
        x = self.unembedding(x)

        # -- auxiliary losses and additional sublayer passes --

        if training and self.config.spliced_loss is not None:
            return x, posterior[-1]
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        return input_shape[:-1] + (
            8 if self.config.predict_repeat_masked else 4,
        )

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self.config.spliced_loss is not None:
                y_pred, posterior = y_pred
                sample_weight = self.spliced_loss_adapter(
                    posterior, sample_weight,
                )
            loss = self.compute_loss(x, y, y_pred, sample_weight)

        self.compute_metrics(x, y, y_pred, sample_weight)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)  # type: ignore
        )
        result = {m.name: m.result() for m in self.metrics}
        result["loss"] = loss
        return result
