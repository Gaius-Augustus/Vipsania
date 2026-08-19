import argparse
from collections.abc import Sequence
from pathlib import Path

DESCRIPTION = (
    "Download a pretrained Vipsania model ahead of time. Annotation "
    "downloads the model it needs on its own, so this is only useful to "
    "prepare a machine that has no internet access when it annotates."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all options of the download command to a parser."""
    parser.add_argument(
        "model_id",
        nargs="+",
        help="one or more model ids; see the table in the Vipsania README",
    )
    parser.add_argument(
        "-d", "--dir",
        default=None,
        help="directory to download into; defaults to the model cache that "
             "the annotation uses as well",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even if the model is already there",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the download with already parsed arguments."""
    from ..hub import cache_dir, download_model, is_available

    root = cache_dir() if args.dir is None else Path(args.dir).expanduser()
    for model_id in args.model_id:
        if not args.force and is_available(model_id, root):
            print(f"Model {model_id} is already in {root/model_id}")
            continue
        download_model(model_id, root=root, force=args.force)

    if args.dir is not None:
        print(
            f"\nAnnotate with these models by adding '--model_dir {root}'."
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the download command."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
