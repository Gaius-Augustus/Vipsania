"""Annotate a genome with a trained Vipsania model.

This is a thin wrapper around the installed command

    $ vipsania annotate ...

so that a cloned repository can be used directly with

    $ python scripts/annotate.py ...
"""

from vipsania.cli.annotate import main

if __name__ == "__main__":
    main()
