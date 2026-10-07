import argparse
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal


def embed_model(
    model: str,
    fasta: str,
    output: Path | str | None = None,
    model_dir: Path | str | None = None,
    weight_name: str = "latest_checkpoint.weights.h5",
    layer: int | list[int] = 7,
    stripe: int | list[int] = 2,
    T: int = 200_000,
    B: int = -1,
    exact: bool = False,
    T_delta: float = 0.1,
    group_limit: int = 1_000_000_000,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    split_seqnames: bool = True,
    dtype: Literal["float32", "float16"] = "float32",
    parallel: int | None = None,
    jit_compile: bool = True,
) -> None:
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    if not model.endswith(".json"):
        from ..hub import model_id_for, resolve
        if model_dir is not None:
            model = model_id_for(model, root=model_dir, download=False)
        else:
            model, model_dir = resolve(model)

    if output is None:
        output = Path(fasta).parent / f"vipsania_{Path(model).stem}.h5"
    output = Path(output).expanduser()

    import vipsania

    from .device import (
        estimate_max_batch_size,
        free_gpu_memory,
        report_devices,
    )
    # measured before TensorFlow allocates anything on the GPU
    free_memory = free_gpu_memory()
    report_devices()

    V = vipsania.create_model(
        model,
        build=True,
        compile=False,
        load=True,
        default_weights_name=weight_name,
        id_parent_folder=model_dir,
    )

    if parallel is None:
        parallel = math.isqrt(T)
        for k in range(parallel, 0, -1):
            if T % k == 0:
                parallel = k
                break
    if T % parallel != 0:
        raise ValueError(
            f"The context length {T} has to be divisible by the degree of "
            f"parallelization {parallel}"
        )
    # the scan of the LRU supports a chunk shorter than 2**(depth+1), so this
    # is one more level than a chunk of length T needs; the extra level costs
    # nothing and keeps a context length that is a power of two working
    lru_tree_depth = (T - 1).bit_length()
    V.set_options(parallel=parallel, tree_depth=lru_tree_depth)

    if B == -1:
        B = estimate_max_batch_size(T, V.count_params(), free_memory)

    from ..embed import _as_list, _resolve_positions
    positions = _resolve_positions(
        V, _as_list(layer, "layer"), _as_list(stripe, "stripe"),
    )
    # several pairs of layer and stripe give one file each, named after it
    outputs = [output] if len(positions) == 1 else [
        output.with_name(f"{output.stem}_l{lid}s{sid}{output.suffix}")
        for lid, sid in positions
    ]

    vipsania.embed_fasta(
        V,
        fasta,
        outputs,
        T=T,
        parallel=parallel,
        layer=layer,
        stripe=stripe,
        B=B,
        exact=exact,
        T_delta=T_delta,
        group_limit=group_limit,
        include=include,
        exclude=exclude,
        split_seqnames=split_seqnames,
        dtype=dtype,
        jit_compile=jit_compile,
        attrs={"model": Path(model).stem},
    )


DESCRIPTION = (
    "Write the residual stream of a Vipsania model at a chosen layer for "
    "every nucleotide of the sequences in a fasta file to an HDF5 file."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all options of the embedding command to a parser."""
    parser.add_argument(
        "model",
        help="path to a model config (.json) or name of a model id (folder)",
    )
    parser.add_argument(
        "fasta",
        help="path to a fasta file (.fa)",
    )

    common = parser.add_argument_group("common")
    common.add_argument(
        "-o", "--output",
        help="output HDF5 file with one dataset of shape (length, d_hidden) "
             "per sequence, named after the sequence; defaults to "
             "'vipsania_[model_id].h5' in the fasta directory",
        default=None,
        type=str,
    )
    common.add_argument(
        "-T", "--context",
        help="genome context length; longer sequences are split into "
             "independently embedded chunks of this length",
        default=200_000,
        type=int,
    )
    common.add_argument(
        "-B", "--batch_size",
        help="batch size for chunks of the full context length",
        default=-1,
        type=int,
    )
    common.add_argument(
        "-l", "--layer",
        help="index of the layer the residual stream is taken from, "
             "starting at 0; defaults to 7; several layers are embedded in "
             "one pass, each into its own file with '_l[layer]s[stripe]' "
             "appended to the name of the output",
        nargs="+",
        default=[7],
        type=int,
    )
    common.add_argument(
        "-s", "--stripe",
        help="index of the stripe within the layer, starting at 0; defaults "
             "to 2, the HMM stripe in the 25M models; give one per layer, or "
             "a single stripe for all layers",
        nargs="+",
        default=[2],
        type=int,
    )
    common.add_argument(
        "--exact",
        help="embed every sequence at its own length instead of padding it "
             "to a shared chunk length; padding with N changes the "
             "embeddings of a sequence, so this is the accurate way to "
             "embed a set of sequences that each fit into one chunk; it "
             "runs without jit, because every length would otherwise be "
             "compiled on its own",
        action="store_true",
    )
    common.add_argument(
        "--dtype",
        help="floating point type the embeddings are stored with",
        choices=["float32", "float16"],
        default="float32",
    )
    common.add_argument(
        "-i", "--include",
        help="only embeds the sequences with the given name",
        nargs="+",
        default=None,
        type=str,
    )
    common.add_argument(
        "-e", "--exclude",
        help="excludes sequences with the given name",
        nargs="+",
        default=None,
        type=str,
    )
    common.add_argument(
        "--keep_seqnames",
        help="do not strip sequence names at the first whitespace character",
        action="store_true",
    )
    common.add_argument(
        "--model_dir",
        help="directory where the model is located; this is searched for "
             "per default",
        default=None,
        type=str,
    )
    common.add_argument(
        "--weights",
        default="latest_checkpoint.weights.h5",
        type=str,
    )

    performance = parser.add_argument_group("performance")
    performance.add_argument(
        "--delta",
        help="sequences are grouped by length and a group of short sequences "
             "is embedded with a chunk length close to its longest sequence; "
             "a sequence is padded to at most 1/delta times its length; "
             "defaults to 0.1",
        type=float,
        default=0.1,
    )
    performance.add_argument(
        "--group_limit",
        help="limits the size of sequence groups to be embedded at once; "
             "defaults to 1,000,000,000",
        type=int,
        default=1_000_000_000,
    )
    performance.add_argument(
        "-p",
        help="sets the degree of parallelization for the HMM",
        default=None,
        type=int,
    )
    performance.add_argument(
        "--nojit",
        help="do not compile the tensorflow model with jit",
        action="store_true",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the embedding with already parsed arguments."""
    embed_model(
        model=args.model,
        fasta=args.fasta,
        output=args.output,
        model_dir=args.model_dir,
        weight_name=args.weights,
        layer=args.layer,
        stripe=args.stripe,
        T=args.context,
        B=args.batch_size,
        exact=args.exact,
        T_delta=args.delta,
        group_limit=args.group_limit,
        include=args.include,
        exclude=args.exclude,
        split_seqnames=not args.keep_seqnames,
        dtype=args.dtype,
        parallel=args.p,
        jit_compile=not (args.nojit or args.exact),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the embedding command."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
