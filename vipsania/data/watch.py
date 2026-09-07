"""Adaptive sampling of training sequences by how much of them is
repeat-masked or unassembled.

Neither repeats nor ``N`` are predicted: repeat positions enter the
masked language modelling loss with the weight ``repeats_loss_weight``
only and ``N`` positions are left out of the masking altogether. A chunk
that consists mostly of either therefore costs a full batch slot for
very little signal, and during a short finetuning it can pull the model
towards whatever those regions look like. Both are summarized in one
content score

    x = repeat fraction + n_weight * N fraction,

so that a single sampling rate governs both and the throughput it
targets means what it says. Dropping every chunk above a fixed score is
the simple remedy, but it is far too strict for repeat-rich genomes and
does nothing at all for clean ones.

Here a chunk of content score ``x`` is instead kept with probability

    a(x) = Phi((mu - x) / sigma) / Phi(mu / sigma),

the survival function of a Gaussian, normalized so that a chunk without
any repeats is always kept. The width ``sigma`` is a hyperparameter;
``sigma -> 0`` recovers the hard filter at ``mu``. The center ``mu`` is
not a hyperparameter but is estimated from the genome itself, as a
quantile of the content score the sampler actually observes, and is
refined while the training runs.
"""

import math
import sys
import time

import numpy as np
import tensorflow as tf
from pydantic import BaseModel

SQRT2 = float(np.sqrt(2.0))

"""Before the first estimate the center sits at `floor`, the fixed
threshold, so that the model never trains on unfiltered sequence while
the estimate is still being gathered. That costs the estimate nothing:
the histogram counts every chunk the filter looks at, before the
decision, so it describes the genome no matter where the center is."""


class RepeatSamplingConfig(BaseModel):

    quantile: float = 0.05
    """Quantile of the observed content score that the acceptance curve
    is centered on, but never below `floor`. Together the two read as:
    keep to the fixed threshold `floor`, and give it up only on a genome
    where fewer than `quantile` of the chunks would survive it.

    The default is low on purpose. On the plant genomes where the fixed
    threshold of 25% clearly helped, between 4% and 8% of the chunks lie
    below it, so at 0.05 the threshold holds and the sampling matches
    what those runs did. A larger value abandons it: at 0.25 the center
    on maize moves to 0.54 and the repeat content of the training mix
    rises from 0.16 to 0.37."""

    n_weight: float = 1.0
    """Weight of the ``N`` fraction in the content score. ``0`` scores
    repeats only and leaves unassembled sequence to
    `drop_N_threshold`."""

    upper_limit_N: float = 0.5
    """Chunks with a larger ``N`` fraction than this are always dropped,
    whatever the curve says. The score cannot tell a chunk that is half
    gap from one that is half repeat, so on a genome where the center
    relaxes far up, a mostly unassembled chunk would otherwise stand a
    real chance of being kept. Such chunks are rare enough that the
    limit costs nothing: of the genomes measured, the most affected was
    an octopus assembly at six accepted chunks in a million."""

    sigma: float = 0.1
    """Width of the acceptance curve, in content score. Small values
    approach a hard filter, large values approach uniform subsampling."""

    floor: float = 0.25
    """The center never falls below this, which is the fixed threshold
    the published models were trained with. Without it the filter would
    be active on every genome, throwing away chunks of a clean one to
    tell 1% repeat content from 3%."""

    refine_every: int = 5_000
    """Chunks the filter looks at between two refinements, counted
    whether they are kept or not. That unit is what the histogram needs
    to estimate a quantile from, and it keeps the refinement going when
    the sampling is slow, which is when the estimate matters most.

    In gradient steps it works out as ``refine_every * keep_rate /
    effective_batch``, so on a genome where a tenth of the chunks
    survive a finetuning refines about every eight steps of the 1000 a
    ten-epoch run takes. Nothing is lost before the first refinement
    either: the center starts at `floor`."""

    decay: float = 0.8
    """How much of the old histogram survives a refinement. The estimate
    keeps a memory of about ``1/(1-decay)`` refinements, which is long on
    purpose: the center should follow the content of the whole genome,
    not of the region that is currently being read."""

    max_step: float = 0.05
    """Largest change of the center in one refinement, so that a run of
    windows from a single repeat-rich locus cannot throw the curve."""

    bins: int = 100
    """Resolution of the histogram the quantile is read from."""

    min_counts: int = 1_000
    """Chunks a genome needs in its own histogram before it stops using
    the pooled histogram of all genomes."""

    verbose: bool = True

    model_config = {"extra": "forbid", "frozen": True}


def _phi(z: float) -> float:
    """The standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(z / SQRT2))


def _quantile(histogram: np.ndarray, q: float) -> float | None:
    """The `q`-quantile of a histogram over the unit interval, or None
    when the histogram is still empty.
    """
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
    """Estimates the repeat content of the genomes that are trained on
    and turns it into the acceptance curve of the dataloader.

    Every genome gets its own histogram and therefore its own curve, so
    that a repeat-rich species in a multi-genome training is not judged
    by the repeat content of the others. Until a genome has seen
    `min_counts` chunks of its own it borrows the pooled histogram.

    The parameters of the curve live in `tf.Variable`s that the filter
    reads for every chunk, which allows refining them while the dataset
    is being consumed, without rebuilding the pipeline.
    """

    def __init__(
        self,
        config: RepeatSamplingConfig | None = None,
        batch_size: int | None = None,
        seed: int = 20_010_101,
    ) -> None:
        self.config = config if config is not None else RepeatSamplingConfig()
        self.batch_size = batch_size
        self.seed = seed
        self.n_genomes = 0
        self.names: list[str] = []
        self.refinements = 0
        self._sigma_warned = False
        self._since = time.monotonic()
        self._last_refine = 0
        self.bind(1)

    def bind(
        self,
        n_genomes: int,
        names: list[str] | None = None,
    ) -> None:
        """Give the watcher the number of genomes it will see. Called by
        the dataset builders once the file list is expanded.
        """
        if names is not None:
            self.names = [str(name) for name in names]
        if n_genomes == self.n_genomes:
            return
        self.n_genomes = n_genomes
        self._histogram = np.zeros((n_genomes, self.config.bins))
        self._fitted = np.zeros(n_genomes, dtype=bool)
        self._was_fitted = np.zeros(n_genomes, dtype=bool)
        with tf.device("CPU:0"):
            self._mu = tf.Variable(
                tf.fill([n_genomes], float(self.config.floor)),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_center",
            )
            self._norm = tf.Variable(
                tf.fill([n_genomes], _phi(
                    self.config.floor / self.config.sigma
                )),
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
            self._kept = tf.Variable(
                tf.zeros([n_genomes]),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_kept",
            )
            # repeats and N of everything seen and of what was kept, so
            # that the report can say what the curve did to each track
            self._sums = tf.Variable(
                tf.zeros([n_genomes, 4]),
                dtype=tf.float32,
                trainable=False,
                name="repeat_sampling_sums",
            )
            self._seen = tf.Variable(
                0,
                dtype=tf.int64,
                trainable=False,
                name="repeat_sampling_seen",
            )

    @property
    def warming_up(self) -> bool:
        """Whether the curve still sits at the fixed threshold, because
        no genome has been estimated yet."""
        return not bool(self._fitted.any())

    @property
    def centers(self) -> np.ndarray:
        """The repeat content each genome's curve is centered on."""
        return self._mu.numpy()

    def accept(
        self,
        repeats: tf.Tensor,
        unassembled: tf.Tensor,
        genome: tf.Tensor | int = 0,
    ) -> tf.Tensor:
        """Decide whether to keep a chunk with the given repeat and `N`
        fraction, and record it. This runs inside the dataset pipeline,
        once per sampled chunk.

        The chunk is counted whatever the decision turns out to be. The
        histogram has to describe the genome, not the chunks that the
        curve lets through; estimating the quantile on the survivors
        alone would walk the center down at every refinement.
        """
        with tf.device("CPU:0"):
            g = tf.cast(genome, tf.int32)
            x = tf.clip_by_value(
                repeats + self.config.n_weight * unassembled, 0.0, 1.0,
            )
            b = tf.clip_by_value(
                tf.cast(x * self.config.bins, tf.int32), 0, self.config.bins-1,
            )
            seen = self._seen.assign_add(1)
            counted = self._counts.scatter_nd_add(
                tf.reshape(tf.stack([g, b]), (1, 2)), tf.ones([1], tf.float32),
            )
            with tf.control_dependencies([counted]):
                mu = tf.gather(self._mu, g)
                norm = tf.gather(self._norm, g)
                a = 0.5 * (
                    1.0 + tf.math.erf((mu - x) / (self.config.sigma * SQRT2))
                ) / norm
                # a stateless draw keyed on the running count, so that the
                # decisions do not repeat and stay independent of how many
                # threads the pipeline happens to use
                nonce = tf.constant(self.seed, tf.int64)
                u = tf.random.stateless_uniform(
                    [], seed=tf.stack([nonce, seen]),
                )
                # a chunk that is almost entirely gap is never useful,
                # not even while the curve still keeps everything
                keep = tf.logical_and(
                    u < a, unassembled <= self.config.upper_limit_N,
                )
            f = tf.cast(keep, tf.float32)
            kept = self._kept.scatter_nd_add(
                tf.reshape(g, (1, 1)), tf.reshape(f, (1,)),
            )
            summed = self._sums.scatter_nd_add(
                tf.reshape(tf.stack([
                    tf.stack([g, 0]), tf.stack([g, 1]),
                    tf.stack([g, 2]), tf.stack([g, 3]),
                ]), (4, 2)),
                tf.stack([
                    repeats, unassembled, f * repeats, f * unassembled,
                ]),
            )
        with tf.control_dependencies([kept, summed]):
            return tf.identity(keep)

    def reset(self) -> None:
        """Restart the clock the sampling rate is measured against. The
        histograms are kept: they describe the genomes, not the run."""
        self._since = time.monotonic()

    def check(self, force: bool = False) -> bool:
        """Refine the estimate if enough chunks were observed since the
        last refinement. Returns whether it was refined.
        """
        seen = int(self._seen.numpy())
        if not force and seen - self._last_refine < self.config.refine_every:
            return False

        fresh = self._counts.numpy()
        self._counts.assign_sub(fresh)
        kept = self._kept.numpy()
        self._kept.assign_sub(kept)
        sums = self._sums.numpy()
        self._sums.assign_sub(sums)
        observed = float(fresh.sum())
        if observed <= 0:
            return False

        self._last_refine = seen
        self._histogram = self.config.decay * self._histogram + fresh
        pooled = self._histogram.sum(axis=0)

        previous = self._mu.numpy()
        self._was_fitted = self._fitted.copy()
        centers = previous.copy()
        estimates = np.full(self.n_genomes, np.nan)
        borrowed = np.zeros(self.n_genomes, dtype=bool)
        for g in range(self.n_genomes):
            own = self._histogram[g]
            if own.sum() < self.config.min_counts:
                own, borrowed[g] = pooled, True
            estimate = _quantile(own, self.config.quantile)
            if estimate is None:
                continue
            estimates[g] = estimate
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
        self._norm.assign(0.5 * (
            1.0 + tf.math.erf(centers / (self.config.sigma * SQRT2))
        ))
        self.refinements += 1

        if self.config.verbose:
            self._report(
                fresh, kept, previous, centers,
                estimates, borrowed, pooled, sums,
            )
        self.reset()
        return True

    def _report(
        self,
        fresh: np.ndarray,
        kept: np.ndarray,
        previous: np.ndarray,
        centers: np.ndarray,
        estimates: np.ndarray,
        borrowed: np.ndarray,
        pooled: np.ndarray,
        sums: np.ndarray,
    ) -> None:
        observed = fresh.sum(axis=1)
        elapsed = max(time.monotonic() - self._since, 1e-9)
        rate = kept.sum() / elapsed
        placed = f"q{100*self.config.quantile:.0f}"
        lines = [
            f"[repeat sampling] refinement {self.refinements}: "
            f"{observed.sum():,.0f} chunks in {elapsed:.1f} s, "
            f"{100*kept.sum()/observed.sum():.0f}% kept ({rate:.1f}/s)"
        ]
        if self.batch_size:
            lines[0] += (
                f", a batch of {self.batch_size} every "
                f"{self.batch_size/max(rate, 1e-9):.2f} s"
            )

        listed = [
            g for g in range(self.n_genomes) if not np.isnan(estimates[g])
        ]
        for g in listed[:8]:
            name = self.names[g] if g < len(self.names) else f"genome {g}"
            note = " (pooled)" if borrowed[g] else ""
            move = centers[g] - previous[g]
            shift = "" if not self._was_fitted[g] else f" ({move:+.3f})"
            limit = " [floor]" if estimates[g] < self.config.floor else ""
            lines.append(
                f"[repeat sampling]   {name}: {placed} = "
                f"{estimates[g]:.3f}{note} -> center {centers[g]:.3f}"
                f"{shift}{limit}"
            )
            lines.append(
                "[repeat sampling]     "
                + self._tracks(sums[g], observed[g], kept[g])
            )
        if len(listed) > 8:
            lines.append(
                f"[repeat sampling]   ... and {len(listed)-8} further "
                f"genomes, centered between {centers[listed].min():.3f} and "
                f"{centers[listed].max():.3f}"
            )
        lines += self._advice(pooled, kept.sum()/observed.sum())
        print("\n" + "\n".join(lines) + "\n", file=sys.stderr, flush=True)

    def _tracks(
        self, sums: np.ndarray, observed: float, kept: float,
    ) -> str:
        """What the curve did to each of the two tracks it scores: the
        mean content of everything that was sampled against the mean
        content of what was kept.
        """
        if observed <= 0:
            return "no chunks"
        if kept <= 0:
            return (
                f"kept nothing of {observed:,.0f} chunks "
                f"(repeats {sums[0]/observed:.0%}, N {sums[1]/observed:.0%})"
            )
        return (
            f"kept {100*kept/observed:.0f}% | "
            f"repeats {sums[0]/observed:.1%} -> {sums[2]/kept:.1%} | "
            f"N {sums[1]/observed:.1%} -> {sums[3]/kept:.1%}"
        )

    def _advice(self, pooled: np.ndarray, keep_rate: float) -> list[str]:
        """Warn when the curve throws chunks away without telling them
        apart, which is a silent failure: the sampling looks like it
        works and only wastes sequences.
        """
        lines = []
        low, high = _quantile(pooled, 0.25), _quantile(pooled, 0.75)
        if low is None or high is None or self._sigma_warned:
            return lines
        spread = high - low
        # a wide sigma is harmless as long as almost everything is kept
        # anyway, which is the usual case on a repeat-poor genome
        if self.config.sigma > max(spread, 1e-6) and keep_rate < 0.9:
            self._sigma_warned = True
            lines.append(
                f"[repeat sampling] !! sigma = {self.config.sigma:.2f} is "
                f"wider than the repeat content of the genomes themselves "
                f"(middle half within {spread:.2f}), so {1-keep_rate:.0%} of "
                f"the sequences are discarded almost regardless of their "
                f"repeats. Lower sigma below {spread:.2f}."
            )
        return lines
