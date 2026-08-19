import argparse
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from timeit import default_timer as clock
from typing import Literal


def _get_available_memory(gpu_index: int = 0) -> float:
    error = (
        "Unable to determine available GPU memory using nvidia-smi. "
        "Specify the batch size manually using '-B' or '--finetune_B'"
    )
    if shutil.which("nvidia-smi") is None: raise RuntimeError(error)

    try:
        result = subprocess.run([
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=10, check=True)
    except Exception as e: raise RuntimeError(error) from e

    output = result.stdout.strip()
    if not output: raise RuntimeError(error)

    try:
        free_mib = float(output.splitlines()[0].strip())
    except ValueError as e:
        raise RuntimeError(error) from e
    return free_mib / 1024


def _estimate_max_batch_size(
    context_length: int,
    model_size_params: int,
    available_memory_gb: float | None = None,
    gpu_index: int = 0,
    safety_factor: float = 0.9,
    finetune: bool = False,
) -> int:
    A = 3.535714285714286e-6 / 14_384_704
    if not finetune:
        # Calibration for 80GB GPU, 25M model, 200k context:
        #   -> batch size: 32
        # Calibration for 24GB GPU, 10M model, 200k context:
        #   -> batch size: 14
        B = 1.125e-5 - 25_047_166 * A
    else:
        # Calibration for 80GB GPU, 25M model, 200k context:
        #   -> batch size: 8
        # Calibration for 24GB GPU, 10M model, 200k context:
        #   -> batch size: 2
        B = 4.5e-5 - 25_047_166 * A

    if available_memory_gb is None:
        available_memory_gb = _get_available_memory(gpu_index=gpu_index)

    usable = safety_factor * available_memory_gb
    cost_per_sample = context_length * (A * model_size_params + B)

    raw_max_batch = usable / cost_per_sample + 1e-9
    if finetune:
        for divisor in (64, 32, 16, 8, 4, 2, 1):
            if divisor <= raw_max_batch: return divisor
    return max(1, int(raw_max_batch))


def _get_genome_size(fasta: str | Path) -> int:
    import bricks2marble as b2m
    return sum(x[3] for x in b2m.io.index(fasta))


def annotate_model(
    model: str,
    fasta: str,
    output: Path | str | None = None,
    model_dir: Path | str | None = None,
    finetune: bool = False,
    finetune_lr: float = 1e-4,
    finetune_B: int = -1,
    finetune_epochs: int = 10,
    T: int = 200_000,
    B: int = -1,
    T_delta: float = 0.1,
    group_limit: int = 1_000_000_000,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    weight_name: str = "latest_checkpoint.weights.h5",
    parallel: int | None = None,
    hmm: int = -1,
    hmm_head: int = 0,
    split_seqnames: bool = True,
    reprediction_factor: float = 0.5,
    repredict_exon_at_boundary: int | None = None,
    penalize_coding_repeats: float | None = None,
    penalize_coding_border_repeats: float | None = None,
    clean: bool = True,
    protein: str | None = None,
    coding: str | None = None,
    alg: Literal["VITERBI", "MEA"] = "VITERBI",
    repeats_input: Literal['track', 'expand', 'omit'] = "track",
    N_token: Literal['track', 'uniform'] = "track",
    drop_repeats_threshold: float | None = None,
    jit_compile: bool = True,
) -> None:
    if output is None:
        output = Path(fasta).parent / f"vipsania_{Path(model).stem}.gff"
    output = Path(output).expanduser()
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    if model_dir is None and not model.endswith(".json"):
        from ..hub import resolve
        model_dir = resolve(model)

    import vipsania

    from .device import report_devices
    report_devices()

    if finetune:
        if finetune_B == -1:
            V = vipsania.create_model(
                model,
                build=True,
                compile=False,
                load=True,
                default_weights_name=weight_name,
                id_parent_folder=model_dir,
                partial_weight_match=penalize_coding_repeats is not None,
                override_config={"model": {"hmm": {
                    "repeats_emitter": penalize_coding_repeats,
                    "repeats_at_borders": penalize_coding_border_repeats,
                }}},
            )
            finetune_B = _estimate_max_batch_size(
                T, V.count_params(), finetune=True,
            )
            import gc

            import tensorflow as tf
            del V
            gc.collect()
            tf.keras.backend.clear_session()

        if finetune_B > 64 or 64 % finetune_B != 0:
            raise ValueError(
                "batch size for finetuning has to be < 64 and "
                f"a divisor of 64; got {finetune_B}"
            )
        if drop_repeats_threshold is None:
            drop_repeats_threshold = (
                0.25 if _get_genome_size(fasta) >= 1e9 else 0.0
            )
        gas = 64 // finetune_B
        trainer = vipsania.Trainer(
            model,
            checkpoints_dir=output.parent,
            offline_checkpoint_prefix="finetuning_",
            model_dir=model_dir,
            jit_compile=jit_compile,
            finetune=True,
            resume=False,
            verbose=True,
            online=None,
            load_weight_name=weight_name,
            override_config={
                "dataset": {
                    "B": finetune_B,
                    "train_paths": [fasta],
                    "indexed_files": True,
                    "indexed_windows_at_once": 1,
                    "validation_paths": None,
                    "drop_repeats_threshold": drop_repeats_threshold,
                },
                "trainer": {
                    "train_steps": 100*gas,
                    "gradient_accumulation_steps": gas,
                    "epochs": finetune_epochs,
                    "lr": finetune_lr,
                    "warmup_steps": None,
                    "decay_steps": None,
                    "spliced_loss_schedule": None,
                    "hyperparameter_schedule": [],
                },
            },
        )
        trainer.create_model()
        finetune_time = clock()
        trainer.train()
        finetune_time = clock() - finetune_time
        V = trainer.model
    else:
        V = vipsania.create_model(
            model,
            build=True,
            compile=False,
            load=True,
            default_weights_name=weight_name,
            id_parent_folder=model_dir,
            partial_weight_match=penalize_coding_repeats is not None,
            override_config={"model": {"hmm": {
                "repeats_emitter": penalize_coding_repeats,
                "repeats_at_borders": penalize_coding_border_repeats,
            }}},
        )

    T_re = int(2*T*reprediction_factor)
    if parallel is None:
        parallel = math.isqrt(T)
        for k in range(parallel, 0, -1):
            if T % k == 0 and T_re % k == 0:
                parallel = k
                break
    lru_tree_depth = (T - 1).bit_length() - 1
    V.set_options(parallel=parallel, tree_depth=lru_tree_depth)

    if B == -1: B = _estimate_max_batch_size(T, V.count_params())

    vipsania.annotate_genome(
        V,
        fasta,
        output,
        T=T,
        parallel=parallel,
        B=B,
        T_delta=T_delta,
        group_limit=group_limit,
        use=alg,
        reprediction_factor=reprediction_factor,
        repredict_exon_at_boundary=repredict_exon_at_boundary,
        include=include,
        exclude=exclude,
        split_seqnames=split_seqnames,
        hmm=hmm,
        hmm_head=hmm_head,
        N_token=N_token,
        repeats_input=repeats_input,
        clean=clean,
        protein_sequence=protein,
        coding_sequence=coding,
        jit_compile=jit_compile,
        logs=[] if not finetune else [
            f"| --- Vipsania finetuning ---",
            f"| learning rate: {finetune_lr}",
            f"| epochs: {finetune_epochs}",
        ] + ([
            f"| maximum repeats: {100*drop_repeats_threshold:.1f}%"
        ] if drop_repeats_threshold is not None and drop_repeats_threshold > 0
          else []
        ) + [
            f"| time: {finetune_time/60:.2f} minutes"
        ],
    )


DESCRIPTION = (
    "Generate a genome annotation of a fasta file with a given trained "
    "Vipsania model."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all options of the annotation command to a parser."""
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
        help="output annotation file (.gtf, .gff or .gp);\n"
             "defaults to 'vipsania_[model_id].gff' in the fasta directory",
        default=None,
        type=str,
    )
    common.add_argument(
        "-T", "--context",
        help="genome context length",
        default=200_000,
        type=int,
    )
    common.add_argument(
        "-B", "--batch_size",
        help="batch size",
        default=-1,
        type=int,
    )
    common.add_argument(
        "-i", "--include",
        help="only annotates the sequences with the given name",
        nargs="+",
        default=None,
        type=str,
    )
    common.add_argument(
        "-e", "--exclude",
        help="excludes sequences with the given name in the annotation",
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

    finetuning = parser.add_argument_group("finetuning")
    finetuning.add_argument(
        "--finetune",
        help="train on the fasta file for a given number of epochs "
             "before annotating; this will save a checkpoint at the output "
             "file location",
        action="store_true",
    )
    finetuning.add_argument(
        "--finetune_lr",
        help="learning rate used for finetuning",
        default=1e-4,
        type=float,
    )
    finetuning.add_argument(
        "--finetune_B",
        help="batch size used for finetuning; only affects GPU memory "
             "consumption, the true effective batch size will be set to 64",
        default=-1,
        type=int,
    )
    finetuning.add_argument(
        "--finetune_epochs",
        help="number of epochs finetuning the model",
        default=10,
        type=int,
    )
    finetuning.add_argument(
        "--drop_repeats",
        help="during finetuning, only train on sequences with low repeat "
             "content; if not specified, will be inferred from the input "
             "genome size",
        default=None,
        type=float,
    )

    add_output = parser.add_argument_group("output")
    add_output.add_argument(
        "--protein",
        help="path to a file where the protein sequence will be extracted to",
        default=None,
        type=str,
    )
    add_output.add_argument(
        "--coding",
        help="path to a file where the coding sequence will be extracted to",
        default=None,
        type=str,
    )

    performance = parser.add_argument_group("performance")
    performance.add_argument(
        "--delta",
        help="a larger delta is helpful for genomes with many small "
             "sequences; defaults to 0.1",
        type=float,
        default=0.1,
    )
    performance.add_argument(
        "--group_limit",
        help="limits the size of sequence groups to be annotated at once; "
             "reduce this for a slower but more memory-friendly annotation; "
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

    quality = parser.add_argument_group("quality")
    quality.add_argument(
        "--rf",
        help="factor in (0, 1]; controls the length of the repredictions",
        default=0.5,
        type=float,
    )
    quality.add_argument(
        "--eb",
        help="forces a reprediction if an exon is near a boundary",
        default=None,
        type=int,
    )
    quality.add_argument(
        "--without_clean",
        help="do not clean the annotation; "
             "e.g. by removing in-frame stop codons",
        action="store_true",
    )
    quality.add_argument(
        "--pcr",
        help="penalize repeats in coding regions with a float between 0 and 1",
        default=None,
        type=float,
    )
    quality.add_argument(
        "--pcbr",
        help="penalize repeats in coding border regions with a float between "
             "0 and 1",
        default=None,
        type=float,
    )
    quality.add_argument(
        "--alg",
        choices=["MEA", "VITERBI"],
        default="VITERBI",
    )

    development = parser.add_argument_group("development")
    development.add_argument(
        "--repeats",
        choices=["track", "expand", "omit"],
        help=argparse.SUPPRESS,
        default="track",
    )
    development.add_argument(
        "--N_token",
        choices=["track", "uniform"],
        help=argparse.SUPPRESS,
        default="track",
    )
    development.add_argument(
        "--hmm",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    development.add_argument(
        "--head",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )


def run(args: argparse.Namespace) -> None:
    """Execute the annotation with already parsed arguments."""
    if args.pcr is not None and args.pcbr is not None:
        raise ValueError(
            "Options pcr and pcbr are not compatible, select only one of them"
        )
    if args.finetune and (args.pcr is not None or args.pcbr is not None):
        raise ValueError(
            "Finetuning combined with options pcr or pcbr is not supported"
        )

    annotate_model(
        model=args.model,
        fasta=args.fasta,
        output=args.output,
        model_dir=args.model_dir,
        finetune=args.finetune,
        finetune_lr=args.finetune_lr,
        finetune_B=args.finetune_B,
        finetune_epochs=args.finetune_epochs,
        weight_name=args.weights,
        T=args.context,
        parallel=args.p,
        B=args.batch_size,
        T_delta=args.delta,
        group_limit=args.group_limit,
        include=args.include,
        exclude=args.exclude,
        hmm=args.hmm,
        hmm_head=args.head,
        split_seqnames=not args.keep_seqnames,
        reprediction_factor=args.rf,
        repredict_exon_at_boundary=args.eb,
        penalize_coding_repeats=args.pcr,
        penalize_coding_border_repeats=args.pcbr,
        clean=not args.without_clean,
        protein=args.protein,
        coding=args.coding,
        alg=args.alg,
        repeats_input=args.repeats,
        N_token=args.N_token,
        drop_repeats_threshold=args.drop_repeats,
        jit_compile=not args.nojit,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the annotation command."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
