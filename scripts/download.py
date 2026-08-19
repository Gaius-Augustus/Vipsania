"""Download a pretrained Vipsania model ahead of time.

This is a thin wrapper around the installed command

    $ vipsania download ...

so that a cloned repository can be used directly with

    $ python scripts/download.py ...
"""

from vipsania.cli.download import main

if __name__ == "__main__":
    main()
