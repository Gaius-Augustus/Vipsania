import sys
import time

import tensorflow as tf

MAX_DISCARD_RATIO = 0.75
"""Fraction of discarded sequences above which the repeat filter is
considered to be the bottleneck. Discarding three out of four sampled
sequences already makes the sampling four times slower than it would be
without the filter."""

RELAXATION_STEP = 0.1
"""Amount by which the allowed repeat content is raised each time the
filter turns out to be too strict for a genome."""

MIN_OBSERVATIONS = 512
"""Number of sequences to look at before judging the sampling."""

MAX_THRESHOLD = 1.0
"""A threshold of one keeps every sequence, so the filter is off."""


class RepeatSamplingWatcher:
    """Counts the sequences that the repeat filter keeps and discards,
    and relaxes the filter when too few sequences get through.

    Repeat-rich genomes can contain so few windows below the allowed
    repeat content that sampling them dominates the runtime. The
    threshold is therefore held in a `tf.Variable` that the filter reads
    for every sequence, which allows raising it while the dataset is
    being consumed, without rebuilding the pipeline.
    """

    def __init__(
        self,
        threshold: float,
        step: float = RELAXATION_STEP,
        max_discard_ratio: float = MAX_DISCARD_RATIO,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        with tf.device("CPU:0"):
            self.threshold = tf.Variable(
                float(threshold),
                dtype=tf.float32,
                trainable=False,
                name="drop_repeats_threshold",
            )
            self._kept = tf.Variable(0, dtype=tf.int64, trainable=False)
            self._discarded = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.initial = float(threshold)
        self.step = step
        self.max_discard_ratio = max_discard_ratio
        self.min_observations = min_observations
        self.relaxations = 0
        self._since = time.monotonic()

    @property
    def value(self) -> float:
        """The repeat content the filter currently allows."""
        return float(self.threshold.numpy())

    @property
    def active(self) -> bool:
        """Whether the filter still removes anything."""
        return self.value < MAX_THRESHOLD

    def observe(self, keep: tf.Tensor) -> tf.Tensor:
        """Count one sequence and return the unchanged filter decision.
        This runs inside the dataset pipeline.
        """
        with tf.device("CPU:0"):
            counted = tf.cast(keep, tf.int64)
            updates = [
                self._kept.assign_add(counted),
                self._discarded.assign_add(1 - counted),
            ]
        with tf.control_dependencies(updates):
            return tf.identity(keep)

    def counts(self) -> tuple[int, int]:
        """The sequences kept and discarded since the last reset."""
        return int(self._kept.numpy()), int(self._discarded.numpy())

    def reset(self) -> None:
        self._kept.assign(0)
        self._discarded.assign(0)
        self._since = time.monotonic()

    def check(self, verbose: bool = True) -> bool:
        """Judge the sampling seen so far and relax the filter if it is
        the bottleneck. Returns whether the threshold was changed.
        """
        kept, discarded = self.counts()
        total = kept + discarded
        if total < self.min_observations:
            return False

        ratio = discarded / total
        rate = kept / max(time.monotonic() - self._since, 1e-9)
        if ratio <= self.max_discard_ratio or not self.active:
            self.reset()
            return False

        previous = self.value
        self.threshold.assign(min(previous + self.step, MAX_THRESHOLD))
        self.relaxations += 1
        if verbose:
            self._warn(ratio, rate, previous)
        self.reset()
        return True

    def _warn(self, ratio: float, rate: float, previous: float) -> None:
        relaxed = (
            "the repeat filter is switched off" if not self.active
            else f"the limit is raised to {self.value:.0%} repeats"
        )
        print(
            f"\n!! The dataloader discarded {ratio:.0%} of the sampled "
            f"sequences ({rate:.1f} usable per second).\n"
            f"!! Sequences with at most {previous:.0%} repeats are rare in "
            f"this genome, so {relaxed}.\n",
            file=sys.stderr,
            flush=True,
        )
