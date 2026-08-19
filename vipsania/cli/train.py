import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

DEFAULT_CHECKPOINTS_DIR = "checkpoints"


def run(args: argparse.Namespace) -> None:
    """Execute the training with already parsed arguments."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
    import vipsania

    from .device import report_devices
    report_devices()

    checkpoints_dir = Path(args.checkpoints).expanduser()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    override_config = {}
    if args.override or args.overrideconfig:
        if args.override is not None:
            override_config = vipsania.util.deep_update(
                override_config, vipsania.util.parse_overrides(args.override),
            )
        if args.overrideconfig is not None:
            with open(args.overrideconfig) as f:
                updates = json.load(f)
            override_config = vipsania.util.deep_update(
                override_config, updates,
            )
        print(f"Applied overrides to model: {override_config}")

    trainer = vipsania.Trainer(
        args.config,
        checkpoints_dir=checkpoints_dir,
        jit_compile=not args.nojit,
        resume=args.resume,
        verbose=int(args.online is None),
        online=args.online,
        override_config=override_config,
    )

    if args.mirrored:
        import tensorflow as tf
        with tf.distribute.MirroredStrategy().scope():
            trainer.create_model()
    else:
        trainer.create_model()

    if args.summary:
        print(trainer.model.summary())
        return

    trainer.train()


DESCRIPTION = (
    "Train a Vipsania model. Supply either a path to a json model "
    "configuration file or the ID for an existing training run."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all options of the training command to a parser."""
    parser.add_argument(
        "config",
        help="path to a model config (.json) or name of a model id (folder)",
    )
    parser.add_argument(
        "-o", "--override",
        action="append",
        default=[],
        help="override config values"
             " (e.g., -o model.ffn_type=glu -o model.ffn.units=128)",
    )
    parser.add_argument(
        "-oc", "--overrideconfig",
        help="path to a config (.json) with arguments to override the given"
             " config; useful when working with --resume",
    )
    parser.add_argument(
        "--checkpoints",
        default=DEFAULT_CHECKPOINTS_DIR,
        help="directory in which the folder of this training run is created;"
             f" defaults to '{DEFAULT_CHECKPOINTS_DIR}' in the current"
             " working directory",
    )
    parser.add_argument(
        "--online",
        type=str,
        default=None,
        help="use the wandb online API to log training progress",
    )
    parser.add_argument(
        "--mirrored",
        action="store_true",
        help="use a mirrored strategy for training on multiple GPUs; "
             "this currently only works with --nojit enabled",
    )
    parser.add_argument(
        "--nojit",
        action="store_true",
        help="deactivate jit compilation of the model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a run from its latest checkpoint",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="only build the model and print its summary, do not train",
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the training command."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
