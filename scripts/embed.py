"""Embed the sequences of a fasta file with a trained Vipsania model.

This is a thin wrapper around the installed command

    $ vipsania embed ...

so that a cloned repository can be used directly with

    $ python scripts/embed.py ...
"""

from vipsania.cli.embed import main

if __name__ == "__main__":
    main()
