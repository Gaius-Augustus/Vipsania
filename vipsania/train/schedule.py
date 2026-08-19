import math

import tensorflow as tf
from hidten.config import with_config
from pydantic import BaseModel


class WarmUpScheduleConfig(BaseModel):

    start_lr: float
    target_lr: float
    warmup_steps: int


@with_config(WarmUpScheduleConfig)  # type: ignore
class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = WarmUpScheduleConfig(**kwargs)

    def __call__(self, step: int) -> float:
        warmup_lr = (
            self.config.start_lr
            + (self.config.target_lr - self.config.start_lr)
            * (tf.cast(step, tf.float32)
               / self.config.warmup_steps)  # type: ignore
        )
        return tf.where(
            step < self.config.warmup_steps,
            warmup_lr,
            self.config.target_lr,
        )


class WarmUpDecayFlatScheduleConfig(BaseModel):

    start_lr: float
    base_lr: float
    warmup_final_lr: float
    warmup_steps: int
    decay_steps: int
    log_decay: int | None = None
    log_velocity: float = 5.0

    cosine: bool = True


@with_config(WarmUpDecayFlatScheduleConfig)  # type: ignore
class WarmUpDecayFlatSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = WarmUpDecayFlatScheduleConfig(**kwargs)

    def __call__(self, step: int) -> float:
        step = tf.cast(step, tf.float32)
        start_lr = tf.cast(self.config.start_lr, tf.float32)
        warmup_final_lr = tf.cast(self.config.warmup_final_lr, tf.float32)
        final_lr = tf.cast(self.config.base_lr, tf.float32)
        warmup_steps = tf.cast(self.config.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.config.decay_steps, tf.float32)
        if self.config.log_decay is not None:
            log_decay_steps = tf.cast(self.config.log_decay, tf.float32)
            log_velocity = tf.cast(self.config.log_velocity, tf.float32)

        def warmup_phase():
            progress = step / warmup_steps
            if self.config.cosine:
                progress = 0.5 * (1 - tf.cos(math.pi * progress))
            return start_lr + (warmup_final_lr - start_lr) * progress

        def decay_phase():
            step_in_decay = step - warmup_steps
            progress = step_in_decay / decay_steps
            if self.config.cosine:
                progress = 0.5 * (1 + tf.cos(math.pi * progress))
                return final_lr + (warmup_final_lr - final_lr) * progress
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            return warmup_final_lr - (warmup_final_lr - final_lr) * progress

        def log_phase():
            step_in = step - warmup_steps - decay_steps
            progress = tf.clip_by_value(step_in / log_decay_steps, 0.0, 1.0)
            decay = tf.math.exp(-log_velocity * progress)
            return final_lr * decay

        def flat_phase():
            return final_lr

        return tf.case([
            (step < warmup_steps, warmup_phase),
            (step < warmup_steps + decay_steps, decay_phase),
        ], default=(
            flat_phase if self.config.log_decay is None else log_phase
        ))


class WarmUpRestartScheduleConfig(BaseModel):

    start_lr: float
    warmup_final_lr: float
    base_lr: float
    warmup_steps: int
    first_cycle_steps: int
    cycle_mult: float = 2.0


@with_config(WarmUpRestartScheduleConfig)  # type: ignore
class WarmUpRestartSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = WarmUpRestartScheduleConfig(**kwargs)

    def post_config_init(self) -> None:
        super().__init__()
        self.cycle = 0

    def __call__(self, step: int) -> float:
        step = tf.cast(step, tf.float32)  # type: ignore
        warmup_steps = tf.cast(self.config.warmup_steps, tf.float32)
        warmup_lr = tf.cast(self.config.warmup_final_lr, tf.float32)
        base_lr = tf.cast(self.config.base_lr, tf.float32)
        min_lr = tf.cast(self.config.start_lr, tf.float32)
        cycle_mult = tf.cast(self.config.cycle_mult, tf.float32)
        first_cycle_steps = tf.cast(self.config.first_cycle_steps, tf.float32)

        def warmup_phase():
            return min_lr + (warmup_lr - min_lr) * (step / warmup_steps)

        def restart_phase():
            step_after_warmup = step - warmup_steps
            n = tf.constant(0, tf.int32)
            total_steps = tf.constant(0.0, tf.float32)
            current_cycle_len = first_cycle_steps

            def cond(n_, total_, cycle_len_):
                return total_ + cycle_len_ <= step_after_warmup

            def body(n_, total_, cycle_len_):
                return n_ + 1, total_ + cycle_len_, cycle_len_ * cycle_mult

            n, total_steps, current_cycle_len = tf.while_loop(
                cond,
                body,
                loop_vars=[n, total_steps, current_cycle_len]
            )  # type: ignore

            step_in_cycle = step_after_warmup - total_steps
            cosine_decay = 0.5 * (
                1 + tf.cos(math.pi * step_in_cycle / current_cycle_len)
            )
            return min_lr + (base_lr - min_lr) * cosine_decay

        return tf.cond(
            step < warmup_steps,
            warmup_phase,
            restart_phase,
        )  # type: ignore
