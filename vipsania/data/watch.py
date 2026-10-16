"""Sampling of training sequences by their repeat and ``N`` content.

A chunk with content score ``x = repeats + n_weight * N`` is kept with
probability

    a(x) = Phi((mu - x) / sigma) / Phi(mu / sigma),

where the center ``mu`` is the `quantile` of the scores observed in the
genome, never below `floor`, and is refined while the training runs.
"""

import math

import numpy as np
import tensorflow as tf
from pydantic import BaseModel

SQRT2 = float(np.sqrt(2.0))


class RepeatSamplingConfig(BaseModel):

    quantile: float = 0.05
    """Quantile of the observed content scores the curve is centered
    on."""
    n_weight: float = 1.0
    """Weight of the ``N`` fraction in the content score."""
    cut_sigmas: float | None = 1.0
    """Chunks more than this many `sigma` above the center are never
    kept; ``None`` keeps the tail of the curve."""
    upper_limit_N: float = 0.5
    """Chunks with a larger ``N`` fraction are always dropped."""
    sigma: float = 0.1
    """Width of the curve; ``0`` is a hard threshold at the center."""
    floor: float = 0.25
    """Lowest possible center and the center before the first
    estimate."""
    refine_every: int = 5_000
    """Sampled chunks between two refinements of the center."""
    decay: float = 0.8
    """Weight of the previous histogram at a refinement."""
    max_step: float = 0.05
    """Largest change of the center in one refinement."""
    bins: int = 100
    min_counts: int = 1_000
    """Chunks a genome needs before its own histogram replaces the one
    pooled over all genomes."""
    verbose: bool = True

    model_config = {"extra": "forbid", "frozen": True}


def _quantile(histogram: np.ndarray, q: float) -> float | None:
    """The `q`-quantile of a histogram over the unit interval, or None
    if it is empty."""
    total = float(histogram.sum())
    if total <= 0:
        return None
    bins = len(histogram)
    cumulated = np.cumsum(histogram)
    index = min(int(np.searchsorted(cumulated, q * total)), bins - 1)
    below = float(cumulated[index - 1]) if index > 0 else 0.0
    height = float(histogram[index])
    inside = 0.0 if height <= 0 else (q * total - below) / height
    return (index + min(max(inside, 0.0), 1.0)) / bins


class RepeatSamplingWatcher:
    """Keeps a histogram of content scores for every genome and turns it
    into that genome's acceptance curve. The curve is held in variables,
    so it can be refined while the dataset is consumed."""

    def __init__(
        self,
        config: RepeatSamplingConfig | None = None,
        seed: int = 20_010_101,
    ) -> None:
        self.config = config if config is not None else RepeatSamplingConfig()
        self.seed = seed
        self.n_genomes = 0
        self.names: list[str] = []
        # set while a progress bar shares the output, which a report then
        # has to start below
        self.progress_bar = False
        self._last_refine = 0
        self.bind(1)

    def bind(
        self,
        n_genomes: int,
        names: list[str] | None = None,
    ) -> None:
        """Set the number of genomes, once the file list is known."""
        if names is not None:
            self.names = [str(name) for name in names]
        if n_genomes == self.n_genomes:
            return
        self.n_genomes = n_genomes
        self._histogram = np.zeros((n_genomes, self.config.bins))
        self._fitted = np.zeros(n_genomes, dtype=bool)
        self._reported = np.full(n_genomes, self.config.floor)
        with tf.device("CPU:0"):
            self._mu = tf.Variable(
                tf.fill([n_genomes], float(self.config.floor)),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_center",
            )
            self._norm = tf.Variable(
                tf.fill([n_genomes], self._normalizer(
                    np.array(self.config.floor, dtype=np.float32)
                ).item()),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_norm",
            )
            self._counts = tf.Variable(
                tf.zeros([n_genomes, self.config.bins]),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_counts",
            )
            self._seen = tf.Variable(
                0,
                dtype=tf.int64,
                trainable=False,
                name="repeat_sampling_seen",
            )

    def _normalizer(self, centers: np.ndarray) -> np.ndarray:
        """Divisor that keeps chunks without repeats at a rate of one."""
        if self.config.sigma <= 0:
            return np.ones_like(centers)
        return 0.5 * (1.0 + np.vectorize(math.erf)(
            centers / (self.config.sigma * SQRT2)
        )).astype(np.float32)

    def accept(
        self,
        repeats: tf.Tensor,
        unassembled: tf.Tensor,
        genome: tf.Tensor | int = 0,
    ) -> tf.Tensor:
        """Whether to keep a chunk with the given repeat and ``N``
        fraction. Runs once per chunk inside the dataset pipeline."""
        with tf.device("CPU:0"):
            g = tf.cast(genome, tf.int32)
            x = tf.clip_by_value(
                repeats + self.config.n_weight * unassembled, 0.0, 1.0,
            )
            b = tf.clip_by_value(
                tf.cast(x * self.config.bins, tf.int32), 0, self.config.bins-1,
            )
            seen = self._seen.assign_add(1)
            # every chunk is counted, not only the kept ones, so that the
            # histogram describes the genome
            counted = self._counts.scatter_nd_add(
                tf.reshape(tf.stack([g, b]), (1, 2)), tf.ones([1], tf.float32),
            )
            with tf.control_dependencies([counted]):
                mu = tf.gather(self._mu, g)
                if self.config.sigma > 0:
                    norm = tf.gather(self._norm, g)
                    a = 0.5 * (1.0 + tf.math.erf(
                        (mu - x) / (self.config.sigma * SQRT2)
                    )) / norm
                else:
                    a = tf.cast(x <= mu, tf.float32)
                # keyed on the running count, so that the draws do not
                # depend on the number of threads of the pipeline
                nonce = tf.constant(self.seed, tf.int64)
                u = tf.random.stateless_uniform(
                    [], seed=tf.stack([nonce, seen]),
                )
                keep = tf.logical_and(
                    u < a, unassembled <= self.config.upper_limit_N,
                )
                if self.config.cut_sigmas is not None:
                    keep = tf.logical_and(
                        keep,
                        x <= mu + self.config.cut_sigmas * self.config.sigma,
                    )
            return keep

    def check(self, force: bool = False) -> bool:
        """Refine the centers if enough chunks were sampled since the last
        refinement. Returns whether they were refined."""
        seen = int(self._seen.numpy())
        if not force and seen - self._last_refine < self.config.refine_every:
            return False

        fresh = self._counts.numpy()
        self._counts.assign_sub(fresh)
        if fresh.sum() <= 0:
            return False

        self._last_refine = seen
        self._histogram = self.config.decay * self._histogram + fresh
        pooled = self._histogram.sum(axis=0)

        previous = self._mu.numpy()
        centers = previous.copy()
        used = []
        for g in range(self.n_genomes):
            own = self._histogram[g]
            if own.sum() < self.config.min_counts:
                own = pooled
            used.append(own)
            estimate = _quantile(own, self.config.quantile)
            if estimate is None:
                continue
            center = max(estimate, self.config.floor)
            if self._fitted[g]:
                center = float(np.clip(
                    center,
                    previous[g] - self.config.max_step,
                    previous[g] + self.config.max_step,
                ))
            centers[g] = center
            self._fitted[g] = True

        self._mu.assign(centers)
        self._norm.assign(self._normalizer(centers))
        if self.config.verbose:
            self._report(centers, used)
        return True

    def _keep_rate(self, histogram: np.ndarray, center: float) -> float:
        """Share of the chunks in a histogram the curve at `center`
        keeps."""
        x = (np.arange(self.config.bins) + 0.5) / self.config.bins
        if self.config.sigma > 0:
            a = 0.5 * (1.0 + np.vectorize(math.erf)(
                (center - x) / (self.config.sigma * SQRT2)
            )) / self._normalizer(np.array(center))
            if self.config.cut_sigmas is not None:
                a[x > center + self.config.cut_sigmas * self.config.sigma] = 0
        else:
            a = (x <= center).astype(float)
        total = histogram.sum()
        return float((histogram * a).sum() / total) if total > 0 else 0.0

    def _report(
        self,
        centers: np.ndarray,
        histograms: list[np.ndarray],
    ) -> None:
        """Print the genomes whose center moved by at least one bin
        since it was last printed."""
        for g in range(self.n_genomes):
            if abs(centers[g] - self._reported[g]) < 1 / self.config.bins:
                continue
            self._reported[g] = centers[g]
            name = self.names[g] if g < len(self.names) else f"genome {g}"
            rate = self._keep_rate(histograms[g], centers[g])
            message = (
                f"[repeat sampling] {name}: center moved to {centers[g]:.2f}, "
                f"keeping about {rate:.0%} of the sequences"
            )
            print(("\n" if self.progress_bar else "") + message, flush=True)
