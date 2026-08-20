"""Download and local storage of pretrained Vipsania models.

A pretrained model is a folder named after its model ID that holds a
configuration and the model weights. Models that are not available
locally are downloaded from :data:`BASE_URL` into :func:`cache_dir` the
first time they are used, and are read from there afterwards.
"""

import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://bioinf.uni-greifswald.de/bioinf/vipsania/models"
"""Location the pretrained models are downloaded from. Overridden by the
environment variable ``VIPSANIA_MODEL_URL``."""

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "latest_checkpoint.weights.h5"
MODEL_FILES = (CONFIG_NAME, WEIGHTS_NAME)
"""The files a pretrained model has to contain. A model folder may hold
more than these, but without them it cannot be used."""

VERSIONS_NAME = "versions.json"
"""Table of the models published per clade, kept next to the models
themselves. It maps a clade name to the model IDs trained for it, the
most recent one last, so that a clade name can be used in place of an
ID. Entries that are not such a list, the date under ``updated`` in
particular, are for the reader and are ignored here."""


def base_url() -> str:
    """The location the pretrained models are downloaded from."""
    return os.environ.get("VIPSANIA_MODEL_URL", BASE_URL).rstrip("/")


def cache_dir() -> Path:
    """The directory downloaded models are stored in.

    This is deliberately not the working directory, so that a model is
    downloaded once and then reused, no matter from where Vipsania is
    called. The location is the first of

        - ``$VIPSANIA_CACHE/models``,
        - ``$XDG_CACHE_HOME/vipsania/models``,
        - ``~/.cache/vipsania/models``.

    Set ``VIPSANIA_CACHE`` if the home directory is not a good place for
    a few hundred megabytes, as is often the case on compute clusters.
    """
    if (path := os.environ.get("VIPSANIA_CACHE")) is not None:
        root = Path(path).expanduser()
    elif (path := os.environ.get("XDG_CACHE_HOME")) is not None:
        root = Path(path).expanduser() / "vipsania"
    else:
        root = Path.home() / ".cache" / "vipsania"
    return root / "models"


def is_available(model_id: str, root: Path | str | None = None) -> bool:
    """Whether all files of `model_id` are already downloaded."""
    path = (cache_dir() if root is None else Path(root)) / model_id
    return all((path / name).is_file() for name in MODEL_FILES)


def versions_path(root: Path | str | None = None) -> Path:
    """Where the clade table is kept locally."""
    return (cache_dir() if root is None else Path(root).expanduser()) / VERSIONS_NAME


def load_versions(
    root: Path | str | None = None,
    download: bool = True,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """The clade table, read locally or fetched from the server.

    An empty table is returned when there is none and none can be
    fetched. A missing table must never stop an annotation, since a
    model named by its ID does not need one.
    """
    path = versions_path(root)
    if download and (refresh or not path.is_file()):
        staging = path.with_name(f".{VERSIONS_NAME}.{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _download_file(f"{base_url()}/{VERSIONS_NAME}", staging, quiet=True)
            os.replace(staging, path)
        except OSError:
            staging.unlink(missing_ok=True)
    try:
        table = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {
        clade: list(ids) for clade, ids in table.items()
        if isinstance(ids, list) and ids
    }


def model_id_for(
    spec: str,
    root: Path | str | None = None,
    download: bool = True,
) -> str:
    """Translate a clade name into the model ID published for it, taking
    the most recent one. Anything that is not a clade name, a model ID
    in particular, is returned unchanged.
    """
    def lookup(table: dict[str, list[str]]) -> str | None:
        for clade, ids in table.items():
            if clade.strip().lower() == spec.strip().lower():
                return ids[-1]
        return None

    found = lookup(load_versions(root, download=False))
    if found is None and download:
        found = lookup(load_versions(root, download=True, refresh=True))
    return found if found is not None else spec


def _report_progress(name: str, read: int, total: int) -> None:
    if total > 0:
        message = (
            f"  {name}: {read/1e6:.1f} / {total/1e6:.1f} MB "
            f"({100*read/total:.0f}%)"
        )
    else:
        message = f"  {name}: {read/1e6:.1f} MB"
    print(f"\r{message}", end="", flush=True)


def _download_file(url: str, target: Path, quiet: bool = False) -> None:
    # Progress is only drawn on a terminal; when the output is a log file,
    # a single line per file is written once the download has finished.
    live = not quiet and sys.stdout.isatty()
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        with open(target, "wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
                read += len(chunk)
                if live: _report_progress(target.name, read, total)
    if live:
        print(flush=True)
    elif not quiet:
        print(f"  {target.name}: {read/1e6:.1f} MB", flush=True)


def _download_archive(url: str, target: Path, quiet: bool = False) -> None:
    """Download a zipped model folder and unpack it, flattening whatever
    directory the archive wraps its files in.
    """
    archive = target / "model.zip"
    _download_file(url, archive, quiet=quiet)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir(): continue
            with zipped.open(member) as source:
                with open(target / Path(member.filename).name, "wb") as file:
                    shutil.copyfileobj(source, file)
    archive.unlink()


def download_model(
    model_id: str,
    root: Path | str | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download the pretrained model `model_id` and return the directory
    that contains its folder.

    The model is first looked for as a folder of individual files and
    then as a zip archive. Nothing is written to `root` until the
    download has completed, so an interrupted download does not leave a
    partial model behind. Several processes may download the same model
    at once, as annotation jobs started together do; whichever finishes
    first publishes its copy and the others keep that one.
    """
    root = cache_dir() if root is None else Path(root).expanduser()
    if not force and is_available(model_id, root):
        return root

    url = f"{base_url()}/{model_id}"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create the model directory {root}. Set the environment "
            "variable VIPSANIA_CACHE to a writable location."
        ) from e

    if not quiet:
        print(f"Downloading Vipsania model {model_id} from {url}", flush=True)
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f".{model_id}."))
    try:
        try:
            for name in MODEL_FILES:
                _download_file(f"{url}/{name}", staging / name, quiet=quiet)
        except urllib.error.HTTPError as e:
            if e.code != 404: raise
            for leftover in staging.iterdir(): leftover.unlink()
            _download_archive(f"{url}.zip", staging, quiet=quiet)

        incomplete = [
            name for name in MODEL_FILES
            if not (staging / name).is_file()
            or (staging / name).stat().st_size == 0
        ]
        if incomplete:
            raise FileNotFoundError(
                f"The model {model_id!r} downloaded from {url} is incomplete: "
                f"{', '.join(incomplete)} missing or empty. Try again, and "
                "report it if the download keeps arriving damaged."
            )

        target = root / model_id
        if force and target.exists():
            shutil.rmtree(target, ignore_errors=True)
        try:
            os.replace(staging, target)
        except OSError:
            # Another process downloaded the same model at the same time and
            # published it first. Its files are the same as ours, so keep them
            # rather than replacing a model that others may already be reading.
            if not is_available(model_id, root):
                raise
    except urllib.error.HTTPError as e:
        raise FileNotFoundError(
            f"No pretrained model {model_id!r} at {base_url()} "
            f"(HTTP {e.code}). Check the name against the table of clades and "
            "model IDs in the Vipsania README. If this is a model you trained "
            "yourself, pass its location with --model_dir."
        ) from e
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach {base_url()} to download the model {model_id!r}: "
            f"{e.reason}. Download the model elsewhere and pass its location "
            "with --model_dir if this machine has no internet access."
        ) from e
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not quiet:
        print(f"Model {model_id} is now in {root/model_id}", flush=True)
    return root


def _find_local(model_id: str) -> Path | None:
    """Search the working directory tree for a model folder, the way
    :func:`vipsania.util.create_model` does.
    """
    from .util import search_folder

    try:
        path = search_folder(Path.cwd(), model_id)
    except NotADirectoryError:
        return None
    return path if (path / CONFIG_NAME).is_file() else None


def resolve(
    spec: str,
    root: Path | str | None = None,
    download: bool = True,
) -> tuple[str, Path | None]:
    """Locate the model named by `spec`, a model ID or a clade name.

    Returns the ID of the model to use together with the directory that
    contains its folder, to be passed to
    :func:`vipsania.util.create_model` as ``id_parent_folder``. That
    directory is ``None`` when the model lies next to the working
    directory, where it is found without any help.
    """
    root = cache_dir() if root is None else Path(root).expanduser()
    if is_available(spec, root):
        return spec, root
    if _find_local(spec) is not None:
        return spec, None

    model_id = model_id_for(spec, root=root, download=download)
    if is_available(model_id, root):
        return model_id, root
    if model_id != spec and _find_local(model_id) is not None:
        return model_id, None
    if not download:
        raise FileNotFoundError(
            f"Neither a model nor a clade named {spec!r} was found in {root}, "
            "and downloading is disabled."
        )
    return model_id, download_model(model_id, root)
