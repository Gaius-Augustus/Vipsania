import csv
import json
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal

import bricks2marble as b2m
import numpy as np
import tensorflow as tf
import wandb
from hidten.config import ModelConfig, T_Model

from ..annotate import annotate_genome
from ..data.watch import RepeatSamplingWatcher
from ..model import Vipsania
from .util import resolve_parameter_path


class EpochSummaryCallback(tf.keras.callbacks.Callback):

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def on_epoch_end(self, epoch, logs=None):
        if self.path.exists():
            with open(self.path) as f:
                summary = json.load(f)
        else:
            summary = {}

        summary["last_epoch"] = epoch + 1
        with open(self.path, "w") as f:
            json.dump(summary, f, indent=4)


class HyperparameterScheduleConfig(ModelConfig):

    parameter: str
    start: float = 1.0
    idle: int | list[int] = 0
    target: float | list[float] = 1.0
    warmup: int | list[int] = 1
    list_index: int | None = None
    verbose: bool = False


class HyperparameterSchedule(tf.keras.callbacks.Callback):

    def __init__(
        self,
        model: tf.keras.Model,
        online: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = HyperparameterScheduleConfig(**kwargs)

        idle = self.config.idle
        warmup = self.config.warmup
        target = self.config.target
        if not isinstance(idle, list): idle = [idle]
        if not isinstance(warmup, list): warmup = [warmup]
        if not isinstance(target, list): target = [target]
        if not (len(idle) == len(warmup) == len(target)):
            raise ValueError("idle, warmup and target must have equal length.")

        self.idle = idle
        self.warmup = warmup
        self.target = target
        self.start = self.config.start

        self.stages = []
        prev_value = self.start
        for k, v, t in zip(self.idle, self.warmup, self.target):
            self.stages.append({
                "idle": k,
                "warmup": v,
                "start_value": prev_value,
                "target_value": t,
            })
            prev_value = t  # next stage starts from previous target

        self.objects = resolve_parameter_path(model, self.config.parameter)
        if not isinstance(self.objects, list): self.objects = [self.objects]
        self.online = online

    def _schedule(self, epoch: int) -> float:
        value = self.start
        for stage in self.stages:
            k = stage["idle"]
            v = stage["warmup"]
            w0 = stage["start_value"]
            t = stage["target_value"]
            if epoch < k:
                return value
            if k <= epoch < k + v:
                alpha = (epoch - k) / float(v)
                return w0 + (t - w0) * alpha
            value = t
        return value

    def on_epoch_begin(self, epoch, logs=None) -> None:
        new_w = self._schedule(epoch)
        for i, obj in enumerate(self.objects):
            if self.config.list_index is None or i == self.config.list_index:
                obj.assign(new_w)
        if self.online:
            key = f"epoch/{self.config.parameter}"
            if self.config.list_index is not None:
                key += f"[{self.config.list_index}]"
            wandb.log({key: new_w})
        if self.config.verbose:
            out = f"-> {self.config.parameter}"
            if self.config.list_index is not None:
                out += f"[{self.config.list_index}]"
            out += f" is now {new_w}"
            print(out, flush=True)


class SplicedLossScheduleConfig(ModelConfig):

    base: str
    idle: int = 1
    each: int = 1
    mult: float = 0.1
    clip: int = 1000
    target: int = 0
    verbose: bool = False
    updates: int | None = None


class SplicedLossSchedule(tf.keras.callbacks.Callback):

    def __init__(
        self,
        model: tf.keras.Model,
        online: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = SplicedLossScheduleConfig(**kwargs)
        self.online = online
        self.conv_kernel_width = resolve_parameter_path(
            model, f"spliced_loss_adapter.conv_kernel_width[{self.config.target}]",
        )
        self.base = resolve_parameter_path(model, self.config.base)
        if self.config.updates is not None: self._updates = 0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        if epoch < self.config.idle: return
        if (epoch-self.config.idle) % self.config.each != 0: return
        if self.config.updates is not None:
            if self._updates == self.config.updates: return
            self._updates += 1

        kernel_width = (
            tf.minimum(self.config.mult / self.base, self.config.clip)
        )
        self.conv_kernel_width.assign(kernel_width)
        if self.online:
            wandb.log({"epoch/spliced_loss_kernel_width": kernel_width})
        if self.config.verbose:
            out = f"spliced loss kernel width is now {kernel_width}"
            print(out, flush=True)


class HookHistory(tf.keras.callbacks.Callback):

    def __init__(
        self,
        hook: tuple[T_Model, str],
        on_epoch: bool = True,
        on_batch: bool = False,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.hook = hook
        self.epoch_history = [] if on_epoch else None
        self.batch_history = [] if on_batch else None
        self.transform = transform

    def on_train_begin(self, logs=None):
        self.hook[0].attach_hooks()

    def on_batch_end(self, batch, logs=None) -> None:
        if self.batch_history is not None:
            hook = self.hook[0].hooks[self.hook[1]]
            if self.transform is not None:
                hook = self.transform(hook)
            self.batch_history.append(hook)

    def on_epoch_end(self, epoch, logs=None) -> None:
        if self.epoch_history is not None:
            hook = self.hook[0].hooks[self.hook[1]]
            if self.transform is not None:
                hook = self.transform(hook)
            self.epoch_history.append(hook)

    def on_train_end(self, logs=None):
        self.hook[0].release_hooks()


class AnnotationMetricsConfig(ModelConfig):

    label: str | None = None

    fasta: Path | str
    reference: Path | str
    gffcompare: Path | str
    include: list[str] | None = None
    T: int = 100_000
    B: int = 32
    group_limit: int = 100_000_000
    split_seqname: bool = True
    gffcompare_e: int = 0

    use_head: int = 0
    use: Literal["VITERBI", "MEA"] = "VITERBI"
    N_token: Literal["track", "uniform"] = "track"
    repeats_input: Literal["track", "expand", "omit"] = "track"

    every_n_epochs: int = 1
    exclude_first_epoch: bool = True
    save_best_key: str | None = None
    jit_compile: bool = True
    translation_table: int | None = None
    log: list[str] | None = [
        "locus/precision", "locus/sensitivity", "locus/F1",
        "intron/precision", "intron/sensitivity", "intron/F1",
        "exon/precision", "exon/sensitivity", "exon/F1",
        "base/precision", "base/sensitivity", "base/F1",
    ]
    csv_name: str = "annotation_metrics.csv"


class AnnotationMetrics(tf.keras.callbacks.Callback):
    """Annotates the given Fasta and logs metrics for the comparison to
    the reference annotation.
    """

    def __init__(
        self,
        save_path: Path,
        online: bool = False,
        **kwargs,
    ) -> None:
        self.model: Vipsania
        self.config = AnnotationMetricsConfig(**kwargs)

        b2m.tools.configure(gffcompare=self.config.gffcompare)
        self.fasta = Path(self.config.fasta)
        self.base_logkey = "annotation"
        if self.config.label is not None:
            self.base_logkey += f"_{self.config.label}"

        self.save_best_key = None
        if self.config.save_best_key is not None:
            self.save_best_key = (
                f"{self.base_logkey}/" + self.config.save_best_key
            )
            self.save_path = save_path / "best_annotation.weights.h5"
            self.best_metric = -1

        self.online = online
        self.csv_path = save_path / self.config.csv_name
        if self.online:
            wandb.define_metric(
                f"{self.base_logkey}/*",
                step_metric="epoch/epoch",
            )

    def write_file(self, epoch: int, metrics: dict[str, float]) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                header = ["epoch"] + list(metrics.keys())
                writer.writerow(header)

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + list(metrics.values()))

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch % self.config.every_n_epochs != 0 or (
            self.config.exclude_first_epoch and epoch == 0
        ):
            return

        metrics = {}
        hmms = list(self.model.get_stripes("hmm"))
        for hmm in range(len(hmms)):

            with tempfile.TemporaryDirectory() as tempdir:
                temp_gp = Path(tempdir) / "vipsania.gp"
                annotate_genome(
                    self.model,
                    self.fasta,
                    temp_gp,
                    self.config.T,
                    parallel=self.model.config.hmm.parallel_factor,
                    hmm=hmm,
                    hmm_head=self.config.use_head,
                    B=self.config.B,
                    include=self.config.include,
                    group_limit=self.config.group_limit,
                    use=self.config.use,
                    N_token=self.config.N_token,
                    repeats_input=self.config.repeats_input,
                    clean=False,
                    jit_compile=self.config.jit_compile,
                    translation_table=self.config.translation_table
                )
                comparison = b2m.tools.compare(
                    temp_gp,
                    self.config.reference,
                    e=self.config.gffcompare_e,
                )

            for key, value in comparison:
                logkey = ""
                if hmm < len(hmms) - 1: logkey += f"hmm{hmm}/"
                logkey += f"{key}"
                f1log = logkey + "/F1"
                if self.config.log is None or f1log in self.config.log:
                    metrics[f"{self.base_logkey}/{f1log}"] = value.F1
                for innerkey, innervalue in value:
                    innerlog = logkey + f"/{innerkey}"
                    if self.config.log is None or innerlog in self.config.log:
                        metrics[f"{self.base_logkey}/{innerlog}"] = innervalue

            if self.save_best_key is not None:
                val = metrics[self.save_best_key]
                if val is not None and self.best_metric < val:
                    self.model.save_weights(
                        self.save_path,
                        overwrite=True,
                    )
                    tf.print(
                        f"Saving model in epoch {epoch}, "
                        f"as {self.save_best_key} improved from "
                        f"{self.best_metric} to {val}."
                    )
                    self.best_metric = val

        self.write_file(
            epoch,
            {k: v for k, v in metrics.items() if not k.startswith("matrix")},
        )
        if self.online:
            wandb.log(metrics)


class WandbHookHistogram(tf.keras.callbacks.Callback):
    """Logs histograms of the given hooks to the wandb API."""

    def __init__(
        self,
        hooks: list[tuple[T_Model, str]],
        data: tf.Tensor,
        every_n_epochs: int = 1,
    ) -> None:
        self.hooks = hooks
        self.data = data
        self.every_n_epochs = every_n_epochs
        wandb.define_metric("hooks/*", step_metric="epoch/epoch")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch % self.every_n_epochs != 0:
            return

        for hook in self.hooks:
            hook[0].attach_hooks()

        self.model(self.data)  # type: ignore

        metrics = {}
        for hook in self.hooks:
            key = f"hooks/{hook[1]}"
            if key in metrics:
                i = 1
                while f"{key}_{i}" in metrics:
                    i += 1
                key = f"{key}_{i}"
            weight = hook[0].hooks[hook[1]].numpy()
            weight[np.isnan(weight)] = 0
            metrics[key] = wandb.Histogram(weight)
        wandb.log(metrics)

        for hook in self.hooks:
            hook[0].release_hooks()
            hook[0].clear_hooks()


class WandbParameterHistogram(tf.keras.callbacks.Callback):
    """Logs each parameter in the model as a histogram to the wandb API.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        ignore_sublabels: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        wandb.define_metric("parameters/*", step_metric="epoch/epoch")
        self.ignore = ignore_sublabels

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch % self.every_n_epochs != 0:
            return

        metrics = {}
        for layer in self.model.layers:  # type: ignore
            for weight in layer.trainable_variables:
                key = f"parameters/{layer.name}/{weight.name}"
                if self.ignore is not None:
                    cont = False
                    for ignore in self.ignore:
                        if ignore in key:
                            cont = True
                    if cont:
                        continue
                if key in metrics:
                    i = 1
                    while f"{key}_{i}" in metrics:
                        i += 1
                    key = f"{key}_{i}"
                if len(weight.shape) == 0:
                    metrics[key] = wandb.Histogram([weight.numpy()])
                    continue
                weight = weight.numpy()
                weight[np.isnan(weight)] = 0
                metrics[key] = wandb.Histogram(weight)
        wandb.log(metrics)


class WandbParameterLine(tf.keras.callbacks.Callback):

    def __init__(
        self,
        param: tf.Variable | Callable[[], tf.Variable],
        label: str,
        every_n_epochs: int = 1,
        islist: bool = False,
        start_at_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.param = param
        self.label = label
        self.every_n_epochs = every_n_epochs
        self.islist = islist
        self.start_at_epoch = start_at_epoch
        wandb.define_metric("matrix/*", step_metric="epoch/epoch")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch % self.every_n_epochs != 0 or epoch < self.start_at_epoch:
            return

        if callable(self.param):
            try:
                param_numpy = self.param()  # type: ignore
            except AttributeError:
                return
        else:
            param_numpy = self.param  # type: ignore
        if len(param_numpy) == 0: return
        if not self.islist:
            wandb.log({f"matrix/{self.label}": param_numpy})
            return
        for i, value in enumerate(param_numpy):
            wandb.log({f"matrix/{self.label}_{i}": value})


class TerminateOnNaNWithCheckpoint(tf.keras.callbacks.TerminateOnNaN):

    def __init__(self, work_dir: Path) -> None:
        super().__init__()
        self.work_dir = work_dir

    def on_batch_end(self, batch, logs=None) -> None:
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                checkpoint_path = self.work_dir / "checkpoint_NaN.weights.h5"
                try:
                    self.model.save_weights(checkpoint_path)
                    print(
                        f"NaN detected in loss at batch index {batch}. "
                        f"Model saved to {checkpoint_path}."
                    )
                except Exception as e:
                    print(f"NaN detected but failed to save checkpoint: {e}")
        super().on_batch_end(batch, logs)


class SamplingMonitor(tf.keras.callbacks.Callback):
    """Watches how many sequences the dataloader has to discard to fill
    a batch and relaxes the repeat filter when the search for usable
    sequences starts to dominate the runtime.

    The check runs on a timer rather than per batch: when the filter is
    far too strict, the first batch of an epoch can take minutes to
    arrive, and that is exactly the situation that has to be noticed.
    """

    def __init__(
        self,
        watcher: RepeatSamplingWatcher,
        interval: float = 30.0,
    ) -> None:
        super().__init__()
        self.watcher = watcher
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _watch(self) -> None:
        while not self._stop.wait(self.interval):
            self.watcher.check()

    def on_train_begin(self, logs=None) -> None:
        self.watcher.reset()
        if self.interval <= 0:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def on_train_end(self, logs=None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval)
            self._thread = None
