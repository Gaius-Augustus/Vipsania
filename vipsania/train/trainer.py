import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import wandb
from pydantic import BaseModel
from wandb.integration.keras import WandbMetricsLogger

from ..data import parallel_files, select_from_indexed_files
from ..data.watch import RepeatSamplingWatcher
from .callback import (AnnotationMetrics, AnnotationMetricsConfig,
                       EpochSummaryCallback, HyperparameterSchedule,
                       HyperparameterScheduleConfig, SamplingMonitor,
                       SplicedLossSchedule, SplicedLossScheduleConfig,
                       TerminateOnNaNWithCheckpoint, WandbHookHistogram,
                       WandbParameterHistogram, WandbParameterLine)


class TrainerConfig(BaseModel):

    epochs: int = 1000
    train_steps: int
    val_steps: int | None = None

    lr: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999
    weight_decay: float = 1e-2
    no_weight_decay: list[str] = ["bias", "gamma", "beta"]
    gradient_clip_norm: float | None = None
    lr_multiplier: dict[str, float | tuple[float, float, float, float]] = {}
    gradient_accumulation_steps: int | None = None

    start_lr: float = 1e-4
    warmup_final_lr: float = 1e-2
    warmup_steps: int | float | None = None
    decay_steps: int | float | None = None
    log_decay_steps: int | float | None = None
    log_decay_velocity: float = 10.0
    lr_cosine: bool = False
    N_cycle: int | None = None
    cycle_factor: float = 2

    spliced_loss_schedule: SplicedLossScheduleConfig | None = None
    hyperparameter_schedule: list[HyperparameterScheduleConfig] = []

    log_annotation_metrics: list[AnnotationMetricsConfig] = []

    resume: str | None = None
    """Set to the ID of an existing run if the training is resumed from
    a previous checkpoint."""

    model_config = {"frozen": True, "extra": "forbid"}


class Trainer:

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: Path | str,
        offline_checkpoint_prefix: str = "version_",
        model_dir: Path | str | None = None,
        jit_compile: bool = True,
        verbose: int = 1,
        online: str | None = None,
        finetune: bool = False,
        resume: bool = False,
        load_weight_name: str = "latest_checkpoint.weights.h5",
        log_param_hist: int = 0,
        log_hooks: int = 0,
        relax_repeats: bool = False,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        override_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.jit_compile = jit_compile
        self.resume = resume
        self.load = finetune if not resume else True
        self.load_weight_name = load_weight_name
        self.checkpoints_dir = Path(checkpoints_dir)
        self.offline_checkpoint_prefix = offline_checkpoint_prefix
        self.model_dir = model_dir
        self.verbose = verbose
        self.online = online
        self._log_freq_hist = log_param_hist
        self._log_freq_hook = log_hooks
        self.relax_repeats = relax_repeats
        self._given_callbacks = callbacks
        self.override_config = override_config
        self.last_epoch = 0
        self.watcher: RepeatSamplingWatcher | None = None

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path

        if self.online is not None:
            self._path = self.checkpoints_dir / wandb.run.id  # type: ignore
            self._path.mkdir()  # type: ignore
        else:
            version = max([0] +
                [int(f.stem[len(self.offline_checkpoint_prefix):])
                 for f in self.checkpoints_dir.iterdir()
                 if f.stem.startswith(self.offline_checkpoint_prefix)]
            ) + 1
            self._path = self.checkpoints_dir / (
                f"{self.offline_checkpoint_prefix}{version}"
            )
            self._path.mkdir()
        return self._path  # type: ignore

    def log(self, config: dict[str, Any], name: str = "config.json") -> None:
        if (cfg_path := self.path/name).exists():
            if not self.resume: raise FileExistsError(
                f"Configuration file {name!r} already exists in {self.path}"
            )
            cfg_path.rename(
                self.path / f"config_epoch_{self.last_epoch:04d}.json"
            )

        with open(self.path / name, "w") as f:
            json.dump(config, f, indent=4)
        if self.online:
            wandb.config.update(config, allow_val_change=True)

    def log_local_copy(self, path: str | Path) -> None:
        path = Path(path)
        (self.path / path.name).write_text(path.read_text())

    def create_model(self) -> None:
        """Initialize, build and compile the model."""
        from ..util import create_model

        self.model, self.config, path = create_model(
            self.model_id,
            build=True,
            compile=True,
            jit_compile=self.jit_compile,
            load=self.load,
            default_weights_name=self.load_weight_name,
            id_parent_folder=self.model_dir,
            verbose=True,
            return_config_and_path=True,
            override_config=self.override_config,
        )

        if self.online is not None:
            entity, project = self.online.split("/")
            wandb.init(
                entity=entity,
                project=project,
                id=path.name if path is not None and self.resume else None,
                resume="must" if self.resume else "never",
            )

        if self.resume:
            if path is not None and (ep := path/"summary.json").exists():
                with open(ep) as f:
                    summary = json.load(f)
                    last_epoch = summary.get("last_epoch", 0)
                self.last_epoch = last_epoch
                print(f"Resuming training after epoch {last_epoch}.")
                with open(ep, "w") as f:
                    resumed = summary.get("resumed_at_epochs", [])
                    resumed.append(last_epoch)
                    summary["resumed_at_epochs"] = resumed
                    json.dump(summary, f, indent=4)
            if path is not None and (
                cp := path / "latest_checkpoint.weights.h5"
            ).exists():
                cp.rename(path / f"epoch_{self.last_epoch:04d}.weights.h5")

        self._path: Path | None = path if self.resume else None

    def _create_datasets(self) -> tuple[
        tf.data.Dataset, tf.data.Dataset | None
    ]:
        """Create training and validation dataset."""
        self.watcher = None
        if (self.relax_repeats
            and self.config.dataset.drop_repeats_threshold > 0
        ):
            self.watcher = RepeatSamplingWatcher(
                self.config.dataset.drop_repeats_threshold,
            )
        if self.config.dataset.indexed_files:
            train_dataset = select_from_indexed_files(
                self.config.dataset.train_paths,
                config=self.config.dataset,
                watcher=self.watcher,
            )
            val_dataset = select_from_indexed_files(
                self.config.dataset.validation_paths,
                config=self.config.dataset,
            ) if self.config.dataset.validation_paths is not None else None
        else:
            train_dataset = parallel_files(
                self.config.dataset.train_paths,
                at_once=self.config.dataset.train_files_at_once,
                config=self.config.dataset,
                watcher=self.watcher,
            )
            val_dataset = parallel_files(
                self.config.dataset.validation_paths,
                at_once=self.config.dataset.validation_files_at_once,
                config=self.config.dataset,
            ) if self.config.dataset.validation_paths is not None else None
        return train_dataset, val_dataset

    def add_callback(self, callback: tf.keras.callbacks.Callback) -> None:
        if self._given_callbacks is None:
            self._given_callbacks = []
        self._given_callbacks.append(callback)

    def _create_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        callbacks = []

        if self.online is not None:
            wandbcallback = WandbMetricsLogger()
            wandb.config.update({"trainable_parameters": sum(
                np.prod(layer.shape)
                for layer in self.model.trainable_weights
            )})
            wandb.config.update({"non_trainable_parameters": sum(
                np.prod(layer.shape)
                for layer in self.model.non_trainable_weights
            )})
            callbacks.append(wandbcallback)

        if (
            self.config.dataset.validation_paths is not None
            and self.config.trainer.val_steps > 0
        ):
            save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
                str(self.path / "best_val_loss.weights.h5"),
                monitor='val_loss',
                verbose=self.verbose,
                save_best_only=True,
                save_weights_only=True,
            )
            callbacks.append(save_best_model_callback)

        if self._log_freq_hist > 0 and self.online is not None:
            callbacks.append(WandbParameterHistogram(
                every_n_epochs=self._log_freq_hist,
                ignore_sublabels=["layer_normalization"],
            ))

        if self._log_freq_hook > 0 and self.online is not None:
            hooks = [(self.model, "embedding")]
            for lid, layer in enumerate(self.model.stripes):
                for sid in range(len(layer)):
                    hooks.append((self.model, f"layer_{lid}_stripe_{sid}"))
                hooks.append((self.model, f"layer_{lid}"))
            callbacks.append(WandbHookHistogram(
                hooks=hooks,
                data=next(iter(self.val_dataset))[0],  # type: ignore
                every_n_epochs=self._log_freq_hook,
            ))

        for lam in self.config.trainer.log_annotation_metrics:
            callbacks.append(AnnotationMetrics(
                save_path=self.path,
                online=self.online is not None,
                **lam.model_dump(),
            ))

        if self.online is not None and (
            self.model.config.spliced_loss is not None
        ):
            callbacks.append(WandbParameterLine(
                self.model.spliced_loss_adapter.memory,
                label="splices",
                every_n_epochs=1,
                islist=True,
                start_at_epoch=0,
            ))

        if self.config.trainer.spliced_loss_schedule is not None:
            callbacks.append(SplicedLossSchedule(
                self.model,
                online=self.online is not None,
                **self.config.trainer.spliced_loss_schedule.model_dump(),
            ))

        for hps in self.config.trainer.hyperparameter_schedule:
            callbacks.append(HyperparameterSchedule(
                model=self.model,
                online=self.online is not None,
                **hps.model_dump(),
            ))

        callbacks.append(EpochSummaryCallback(path=self.path/"summary.json"))

        if self._given_callbacks is not None:
            callbacks += self._given_callbacks

        return callbacks

    def _create_essential_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        callbacks = []
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(self.path / "latest_checkpoint.weights.h5"),
            monitor='loss',
            verbose=self.verbose,
            save_best_only=False,
            save_weights_only=True,
        ))
        callbacks.append(TerminateOnNaNWithCheckpoint(work_dir=self.path))
        if self.watcher is not None:
            callbacks.append(SamplingMonitor(self.watcher))
        return callbacks

    def train(self) -> None:
        self.log(self.config.model_dump(), name="config.json")
        self.train_dataset, self.val_dataset = self._create_datasets()
        callbacks = self._create_essential_callbacks()
        if self.resume or not self.load: callbacks += self._create_callbacks()
        self.model.fit(
            self.train_dataset,
            callbacks=callbacks,
            epochs=self.config.trainer.epochs,
            initial_epoch=self.last_epoch,
            steps_per_epoch=self.config.trainer.train_steps,
            validation_data=self.val_dataset,
            validation_steps=self.config.trainer.val_steps,
            verbose=self.verbose,  # type: ignore
        )

        if self.online is not None:
            wandb.finish()

    def test(self) -> None:
        self.train_dataset, _ = self._create_datasets()
        self.model.evaluate(
            self.train_dataset,
            steps=self.config.trainer.train_steps,
            verbose=self.verbose,  # type: ignore
        )
