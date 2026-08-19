import re
from typing import TYPE_CHECKING, Any

import tensorflow as tf

from .metrics import MaskedAccuracy, MaskedPerplexity
from .schedule import (WarmUpDecayFlatSchedule, WarmUpRestartSchedule,
                       WarmUpSchedule)

if TYPE_CHECKING:
    from .trainer import TrainerConfig


def tf_schedule(base, idle, target, warmup):
    base = tf.convert_to_tensor(base, tf.float32)
    target = tf.convert_to_tensor(target, tf.float32)
    idle = tf.cast(idle, tf.float32)
    warmup = tf.cast(warmup, tf.float32)

    def schedule(step):
        step = tf.cast(step, tf.float32)
        ratio = (step - idle) / warmup
        ratio = tf.clip_by_value(ratio, 0.0, 1.0)
        return base + (target - base) * ratio

    return schedule


class MultiLRAdamW(tf.keras.optimizers.AdamW):

    def __init__(
        self,
        #                                tuple: base, idle, target, warmup
        lr_multiplier: dict[str, float | tuple[float, float, float, float]],
        train_steps: int,
        epochs: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lr_multiplier = {}
        for key, value in lr_multiplier.items():
            if isinstance(value, (float, int)):
                self.lr_multiplier[key] = lambda _: value
            else:
                base, idle, target, warmup = value
                idle *= train_steps * epochs
                warmup *= train_steps * epochs
                self.lr_multiplier[key] = tf_schedule(
                    base, idle, target, warmup,
                )

    def apply_gradients(self, grads_and_vars, **kwargs):
        scaled = []
        for grad, var in grads_and_vars:
            if grad is not None:
                scale = 1.0
                for key, multiplier in self.lr_multiplier.items():
                    if key in var.path:
                        scale = tf.cast(
                            multiplier(self.iterations),
                            tf.float32,
                        )
                        break

                if isinstance(grad, tf.IndexedSlices):
                    grad = tf.IndexedSlices(
                        values=grad.values * scale,
                        indices=grad.indices,
                        dense_shape=grad.dense_shape
                    )
                else:
                    grad = grad * scale

            scaled.append((grad, var))
        return super().apply_gradients(scaled, **kwargs)


def p_cycle(p_warmup: float, N_cycle: int, cycle_factor: float) -> float:
    return (cycle_factor - 1) * (1 - p_warmup) / (cycle_factor**N_cycle - 1)


def compile_model(
    model: tf.keras.Model,
    config: "TrainerConfig",
    jit_compile: bool = False,
) -> tf.keras.Model:
    lr = config.lr
    if config.warmup_steps is not None:
        ws = config.warmup_steps
        if 0.0 <= ws <= 1.0:
            ws *= config.train_steps * config.epochs
        else:
            ws *= config.train_steps
        ws = int(ws)
        if config.N_cycle is not None:
            cs = p_cycle(
                config.warmup_steps,
                config.N_cycle,
                config.cycle_factor,
            )
            cs = int(cs * config.train_steps * config.epochs)
            lr = WarmUpRestartSchedule(
                start_lr=config.start_lr,  # type: ignore
                warmup_final_lr=config.warmup_final_lr,  # type: ignore
                base_lr=config.lr,  # type: ignore
                warmup_steps=ws,  # type: ignore
                first_cycle_steps=cs,  # type: ignore
                cycle_mult=config.cycle_factor,  # type: ignore
            )
        elif config.decay_steps is not None:
            ds = config.decay_steps
            if 0.0 <= ds <= 1.0:
                ds *= config.train_steps * config.epochs
            else:
                ds *= config.train_steps
            ds = int(ds)
            ls = config.log_decay_steps
            if ls is not None and 0.0 <= ls <= 1.0:
                ls *= config.train_steps * config.epochs
            elif ls is not None:
                ls *= config.train_steps

            if ls is not None:
                ls = int(ls)
            lr = WarmUpDecayFlatSchedule(
                start_lr=config.start_lr,  # type: ignore
                base_lr=config.lr,  # type: ignore
                warmup_final_lr=config.warmup_final_lr,  # type: ignore
                warmup_steps=ws,  # type: ignore
                decay_steps=ds,  # type: ignore
                log_decay=ls,  # type: ignore
                log_velocity=config.log_decay_velocity,  # type: ignore
                cosine=config.lr_cosine,  # type: ignore
            )
        else:
            lr = WarmUpSchedule(
                start_lr=config.start_lr,  # type: ignore
                target_lr=config.lr,  # type: ignore
                warmup_steps=ws,  # type: ignore
            )

    optimizer = MultiLRAdamW(
        lr_multiplier=config.lr_multiplier,
        train_steps=config.train_steps,
        epochs=config.epochs,
        learning_rate=lr,  # type: ignore
        weight_decay=config.weight_decay,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        clipnorm=config.gradient_clip_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    optimizer.exclude_from_weight_decay(var_names=config.no_weight_decay)
    optimizer.build(model.trainable_variables)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        ignore_class=-1,
    )
    metrics = [
        MaskedAccuracy(name="accuracy"),
        MaskedPerplexity(name="perplexity"),
    ]
    model.compile(
        optimizer=optimizer,  # type: ignore
        loss=loss,
        jit_compile=jit_compile,  # type: ignore
        metrics=metrics,
    )
    return model


def resolve_parameter_path(
    model: tf.keras.Model,
    parameter: str,
) -> Any:
    """Traverse a model graph and return all paramters and layers
    accessed by a given string. Allowed is also a list-like access
    using either indices or wildcards such as `layer[0].mlp.kernel` or
    `layer[*].mlp.kernel`.
    """
    parts = parameter.split(".")
    current_objects = [model]

    for part in parts:
        next_objects = []
        match = re.match(r"(\w+)(?:\[(\*|\d+)\])?$", part)
        if not match: raise ValueError(f"Invalid variable name given: {part}")

        attr_name, index = match.groups()
        for o in current_objects:
            o = getattr(o, attr_name)

            if index is None:
                next_objects.append(o)
            elif index == "*":
                next_objects.extend(o)
            else:
                next_objects.append(o[int(index)])

        current_objects = next_objects

    if len(current_objects) == 1:
        return current_objects[0]
    return current_objects
