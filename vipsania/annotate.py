from collections.abc import Container
from functools import partial
from pathlib import Path
from typing import Literal

import bricks2marble as b2m
import numpy as np
from hidten import HMMMode

from .model.base import Vipsania
from .xai.evaluate import predict_sequence


def _fix_intron_state_chain_labels(a: np.ndarray, isc: int) -> np.ndarray:
    mask = np.logical_and(0 < a, a <= 3*isc)
    a[mask] = ((a[mask] - 1) % 3) + 1
    a[a > 3*isc] = a[a > 3*isc] - 3*(isc-1)
    return a


def _evaluate(
    fasta: b2m.struct.Fasta,
    model: Vipsania,
    B_ref_T: tuple[int, int],
    hmm: int = -1,
    hmm_head: int = 0,
    use: Literal["VITERBI", "MEA", "POSTERIOR_SAMPLE"] = "VITERBI",
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    jit_compile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    B = int(np.floor(B_ref_T[0] * B_ref_T[1] / fasta.T))
    model.toggle_inference(HMMMode[use], hmm=hmm, enable=True)
    labels = predict_sequence(
        model,
        fasta,
        B=B,
        return_batched=True,
        N_token=N_token,
        repeats_input=repeats_input,
        jit_compile=jit_compile,
    ).numpy()  # type: ignore
    model.toggle_inference(HMMMode[use], hmm=hmm, enable=False)
    H = labels.shape[2] // 2
    labels_f = labels[:, :, hmm_head]
    labels_b = labels[:, :, H+hmm_head]
    if (isc := list(model.get_stripes("hmm"))[0].config.hmm.intron_state_chain
    ) > 1:
        labels_f = _fix_intron_state_chain_labels(labels_f, isc=isc)
        labels_b = _fix_intron_state_chain_labels(labels_b, isc=isc)
    return labels_f, labels_b


def annotate_genome(
    model: Vipsania,
    fasta: Path | str,
    output: Path | str,
    T: int,
    parallel: int,
    hmm: int = -1,
    hmm_head: int = 0,
    B: int = 1,
    T_delta: float = 0.1,
    group_limit: int = 1_000_000_000,
    include: Container[str] | None = None,
    exclude: Container[str] | None = None,
    split_seqnames: bool = True,
    use: Literal["VITERBI", "MEA", "POSTERIOR_SAMPLE"] = "VITERBI",
    reprediction_factor: float = 0.5,
    repredict_exon_at_boundary: int | None = None,
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    clean: bool = True,
    protein_sequence: str | None = None,
    coding_sequence: str | None = None,
    jit_compile: bool = False,
    logs: list[str] | None = None,
    translation_table: int | None = None,
) -> None:

    def post(
        fasta: b2m.struct.Fasta,
        annotation: b2m.struct.Annotation,
    ) -> b2m.struct.Annotation:
        b2m.tools.check_min_coding_length(annotation, length=9, remove=True)
        b2m.tools.check_inframe_stop_codons(annotation, fasta, remove=True, translation_table=translation_table)
        if protein_sequence is not None:
            annotation.sequence_to_file(
                "protein",
                fasta,
                protein_sequence,
                line_width=80,
                mode="a",
                translation_table=translation_table,
            )
        if coding_sequence is not None:
            annotation.sequence_to_file(
                "coding",
                fasta,
                coding_sequence,
                line_width=80,
                mode="a",
            )
        return annotation

    b2m.tools.annotate_genome(
        fasta,
        predict_func=partial(_evaluate,
            model=model,
            B_ref_T=(B, T),
            hmm=hmm,
            hmm_head=hmm_head,
            use=use,
            N_token=N_token,
            repeats_input=repeats_input,
            jit_compile=jit_compile,
        ),
        output=output,
        allow_extract_gz=True,
        T_max=T,
        T_delta=T_delta,
        T_factors=[parallel],
        group_size_limit=group_limit,
        model_name="Vipsania",
        include_seqs=include,
        exclude_seqs=exclude,
        split_seqnames=split_seqnames,
        reprediction_factor=reprediction_factor,
        repredict_exon_at_boundary=repredict_exon_at_boundary,
        postprocess=post if clean else None,
        log_config=[
            f"Vipsania total parameters: {model.count_params()}",
            f"batch size for maximal chunk length: {B}",
        ] + (logs if logs is not None else []),
    )
