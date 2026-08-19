"""Train a Vipsania model.

This is a thin wrapper around the installed command

    $ vipsania train ...

so that a cloned repository can be used directly with

    $ python scripts/train.py ...
"""

from vipsania.cli.train import main

if __name__ == "__main__":
    main()
