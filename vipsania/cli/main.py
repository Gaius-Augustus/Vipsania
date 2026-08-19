import argparse
from typing import Sequence

from . import annotate, download, train

COMMANDS = {
    "annotate": annotate,
    "train": train,
    "download": download,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``vipsania`` command."""
    parser = argparse.ArgumentParser(
        prog="vipsania",
        description="Vipsania - an unsupervised deep learning ab-initio gene "
                    "finder.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, module in COMMANDS.items():
        subparser = commands.add_parser(
            name,
            help=module.DESCRIPTION,
            description=module.DESCRIPTION,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(run=module.run)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
