import ast
import json
import warnings
from pathlib import Path
from typing import Any, Literal, overload

from pydantic import BaseModel

from .data.dataset import DatasetConfig
from .model.base import Vipsania, VipsaniaConfig
from .train.trainer import TrainerConfig
from .train.util import compile_model


class RunConfig(BaseModel):

    model: VipsaniaConfig
    dataset: DatasetConfig
    trainer: TrainerConfig


def search_folder(
    path: Path,
    name: Path | str,
    above: int = 2,
    below: int = 2,
) -> Path:
    found = []

    def _search(path: Path, height: int = 0) -> None:
        def _below(path: Path, depth: int = 0) -> None:
            if depth == below:
                return
            try:
                for thing in path.iterdir():
                    if thing.is_dir():
                        if thing.name == str(name): found.append(thing)
                        else: _below(thing, depth+1)
            except PermissionError:
                pass
        if height == above:
            return
        try:
            for thing in path.parent.iterdir():
                if thing.is_dir():
                    if thing.name == str(name): found.append(thing)
                    else: _below(thing)
        except PermissionError:
            pass
        _search(path.parent, height+1)

    _search(path/"any")
    found = list(set(found))
    if len(found) == 0:
        raise NotADirectoryError(f"Did not find a directory named '{name!s}'.")
    if len(found) > 1:
        warnings.warn(
            f"More than one folder with name '{name!s}' found, "
            "returning the first occurence.",
        )
    return found[0]


def deep_update(
    base: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    """Replace entries of the possibly nested base dictionary that are
    inside the override dictionary. Inner dictionaries are also updated
    and not overwritten.
    """
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_runconfig(
    file: Path | str,
    override_model: dict[str, Any] | None = None,
    override_dataset: dict[str, Any] | None = None,
    override_trainer: dict[str, Any] | None = None,
    resume: str | None = None,
    translation_table: int | None = None,
) -> RunConfig:
    with open(file, "r") as f:
        total_dict = json.load(f)
    if (
        "model" not in total_dict
        or "dataset" not in total_dict
        or "trainer" not in total_dict
    ):
        raise ValueError(
            "A total config has to consist of model, dataset and trainer."
        )

    if override_model is not None:
        total_dict["model"] = deep_update(total_dict["model"], override_model)
    if override_dataset is not None:
        total_dict["dataset"] = deep_update(
            total_dict["dataset"],
            override_dataset,
        )
    if override_trainer is not None:
        total_dict["trainer"] = deep_update(
            total_dict["trainer"],
            override_trainer,
        )

    if resume is not None:
        total_dict["trainer"]["resume"] = resume

    if total_dict["model"].get("hmm") is not None:
        total_dict["model"]["hmm"]["translation_table"] = translation_table

    return RunConfig(**total_dict)


@overload
def create_model(
    config_or_id: str,
    /,
    build: bool | tuple[None | int, ...] = ...,
    compile: bool = ...,
    jit_compile: bool = ...,
    load: bool = ...,
    weights: Path | str | None = ...,
    partial_weight_match: bool = ...,
    default_config_name: str = ...,
    default_weights_name: str = ...,
    override_config: dict[str, Any] | None = ...,
    id_search_depth: int = ...,
    id_search_height: int = ...,
    id_parent_folder: Path | str | None = ...,
    verbose: bool = ...,
    return_config_and_path: Literal[False] = ...,
    translation_table: int | None = None,
) -> Vipsania:
    ...
@overload
def create_model(
    config_or_id: str,
    /,
    build: bool | tuple[None | int, ...] = ...,
    compile: bool = ...,
    jit_compile: bool = ...,
    load: bool = ...,
    weights: Path | str | None = ...,
    partial_weight_match: bool = ...,
    default_config_name: str = ...,
    default_weights_name: str = ...,
    override_config: dict[str, Any] | None = ...,
    id_search_depth: int = ...,
    id_search_height: int = ...,
    id_parent_folder: Path | str | None = ...,
    verbose: bool = ...,
    return_config_and_path: Literal[True] = ...,
    translation_table: int | None = None,
) -> tuple[Vipsania, RunConfig, Path | None]:
    ...
def create_model(
    config_or_id: str,
    /,
    build: bool | tuple[None | int, ...] = True,
    compile: bool = True,
    jit_compile: bool = False,
    load: bool = True,
    weights: Path | str | None = None,
    partial_weight_match: bool = False,
    default_config_name: str = "config.json",
    default_weights_name: str = "latest_checkpoint.weights.h5",
    override_config: dict[str, Any] | None = None,
    id_search_depth: int = 4,
    id_search_height: int = 3,
    id_parent_folder: Path | str | None = None,
    verbose: bool = True,
    return_config_and_path: bool = False,
    translation_table: int | None = None,
) -> Vipsania | tuple[Vipsania, RunConfig, Path | None]:
    """Creates a `Vipsania` model with a given configuration file or
    model ID.
    The model ID is a name of a folder (that is searched for
    automatically with the options `id_search_depth` and
    `id_search_height`). The folder should contain a "config.json" file
    that consists of the keys "model", "dataset" and "trainer". To avoid
    ambiguity, it is also possible to pass the `id_parent_folder`, which
    is then used instead of performing a search.

    The default is that the model will be built and compiled
    automatically if the given string is a model ID. Optionally, weights
    in this folder can then be loaded.
    """
    if config_or_id.endswith(".json"):
        config_file = config_or_id
        path = Path(config_file).parent
    else:
        if id_parent_folder is not None:
            path = Path(id_parent_folder).expanduser() / config_or_id
        else:
            path = search_folder(
                Path.cwd(),
                config_or_id,
                below=id_search_depth,
                above=id_search_height,
            )
        if verbose: print(
            f"Found model location {path.resolve()}"
        )
        config_file = path / default_config_name
        if not config_file.exists():
            raise FileNotFoundError(
                f"File {default_config_name!r} does not exist for "
                f"given id {config_file!r}"
            )
        if verbose: print(
            f"Using model config {config_file.resolve()}"
        )

    override_model = None
    override_dataset = None
    override_trainer = None
    if override_config is not None:
        override_model = override_config.get("model")
        override_dataset = override_config.get("dataset")
        override_trainer = override_config.get("trainer")
    config = load_runconfig(
        config_file,
        override_model=override_model,
        override_dataset=override_dataset,
        override_trainer=override_trainer,
        resume=config_or_id if load else None,
        translation_table=translation_table,
    )

    model = Vipsania(**config.model.model_dump())
    if verbose: print(f"Model successfully created.")

    if isinstance(build, tuple):
        model.build(build)
        if verbose: print(f"Model successfully built.")
    elif build:
        model.build((None, None, config.dataset.input_dim))
        if verbose: print(f"Model successfully built.")

    if compile:
        model = compile_model(model, config.trainer, jit_compile=jit_compile)
        if verbose: print(f"Model successfully compiled.")
    elif jit_compile:
        model.compile(jit_compile=jit_compile)  # type: ignore

    if load:
        if weights is None:
            if path is None: raise RuntimeError("Model weights are missing.")
            weights = path / default_weights_name
            if verbose: print(
                f"Using model weights {weights.resolve()}"
            )
        if partial_weight_match:
            try:  # by_name is only supported in older Keras versions
                model.load_weights(
                    weights,
                    skip_mismatch=partial_weight_match,
                    by_name=True,
                )
            except (TypeError, ValueError):
                model.load_weights(weights, skip_mismatch=partial_weight_match)
        else:
            model.load_weights(weights, skip_mismatch=partial_weight_match)

        if verbose: print(f"Model weights successfully loaded.")

    if return_config_and_path:
        return model, config, path
    return model


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse command-line override strings into nested dictionaries.

    Examples:
        ["model.ffn_type=glu"] -> {"model": {"ffn_type": "glu"}}
        ["model.hmm_at=[0]"] -> {"model": {"hmm_at": [0]}}
        ["model.hmm.heads=2"] -> {"model": {"hmm": {"heads": 2}}}
    """
    result = {}

    for item in pairs:
        if '=' not in item: continue

        key, value_str = item.split('=', 1)
        keys = key.split('.')

        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str

        current = result
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return result
