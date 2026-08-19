import tensorflow as tf


def masked_accuracy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
    ignore_label: int = -1,
) -> tf.Tensor:
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)  # type: ignore

    mask = tf.not_equal(y_true, ignore_label)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    return tf.cast(tf.equal(y_true_masked, y_pred_masked), y_pred.dtype)


class MaskedAccuracy(tf.keras.metrics.Metric):

    def __init__(
        self,
        name: str = "accuracy",
        ignore_label: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.ignore_label = ignore_label
        self.total = self.add_weight(initializer="zeros")
        self.count = self.add_weight(initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        matches = masked_accuracy(
            y_true, y_pred,
            sample_weight=sample_weight,
            ignore_label=self.ignore_label,
        )
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(tf.size(matches), self.dtype))

    def result(self) -> tf.Tensor:
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_states(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)


def masked_perplexity(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    sample_weight: tf.Tensor | None = None,
    ignore_label: int = -1,
) -> tf.Tensor:
    mask = tf.not_equal(y_true, ignore_label)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    sparse_crossentropy = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_masked,
        y_pred_masked,
        from_logits=True,
    )
    return tf.cast(sparse_crossentropy, dtype=y_pred.dtype)


class MaskedPerplexity(tf.keras.metrics.Metric):

    def __init__(
        self,
        name: str = "perplexity",
        ignore_label: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.ignore_label = ignore_label
        self.total = self.add_weight(initializer="zeros")
        self.count = self.add_weight(initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        perplexity = masked_perplexity(
            y_true, y_pred,
            sample_weight=sample_weight,
            ignore_label=self.ignore_label,
        )
        self.total.assign_add(tf.reduce_sum(perplexity))
        self.count.assign_add(tf.cast(tf.size(perplexity), self.dtype))

    def result(self) -> tf.Tensor:
        return 2 ** (-self.total / (self.count + tf.keras.backend.epsilon()))

    def reset_states(self) -> None:
        self.total.assign(0.0)
        self.count.assign(0.0)
