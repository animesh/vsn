#python run_and_compare.py "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv" --python-output "python_vsn_matched_samples.tsv"
"""Generate Python VSN2 values and compare them with an R VSN output.
Keep this file beside run.py, compare.py, and vsn2.py.
"""

import argparse
from pathlib import Path
import pandas as pd
from compare import (
    R_NON_INTENSITY_COLUMNS,
    canonical_sample,
    compare_outputs,
    infer_separator,
    numeric_candidate_columns,
)
from run import run_vsn2_matrix

DEFAULT_ID_COLUMNS = [
    "Protein.Group",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
    "N.Sequences",
    "N.Proteotypic.Sequences",
]


def unique_map(columns, label):
    result = {}
    for column in columns:
        key = canonical_sample(column)
        if key in result:
            raise ValueError(
                f"Duplicate canonical {label} sample {key!r}: "
                f"{result[key]!r} and {column!r}"
            )
        result[key] = column
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select the samples represented in an R VSN result, generate "
            "Python VSN2 values, save them, and compare both outputs."
        )
    )
    parser.add_argument("raw_matrix", help="Original raw CSV/TSV matrix")
    parser.add_argument("r_output", help="R VSN CorTestBH CSV/TSV")
    parser.add_argument(
        "--python-output",
        default="python_vsn_matched_samples.tsv",
        help="Reusable Python VSN2 output for later comparison",
    )
    parser.add_argument(
        "--details-output",
        default="vsn_matched_comparison_by_sample.csv",
    )
    parser.add_argument("--differences-output")
    parser.add_argument("--parameters-output")
    parser.add_argument("--raw-separator", help=r"Use \t for tab")
    parser.add_argument("--r-separator", help=r"Use \t for tab")
    parser.add_argument("--lts-quantile", type=float, default=0.9)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    raw_separator = infer_separator(args.raw_matrix, args.raw_separator)
    r_separator = infer_separator(args.r_output, args.r_separator)

    raw = pd.read_csv(
        args.raw_matrix,
        sep=raw_separator,
        encoding="utf-8-sig",
        low_memory=False,
    )
    r_result = pd.read_csv(
        args.r_output,
        sep=r_separator,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(raw) != len(r_result):
        raise ValueError(
            f"Row-count mismatch: raw matrix has {len(raw)} rows; "
            f"R result has {len(r_result)} rows"
        )

    raw_numeric = numeric_candidate_columns(raw, excluded=set())
    r_numeric = numeric_candidate_columns(
        r_result,
        excluded=R_NON_INTENSITY_COLUMNS,
    )
    raw_map = unique_map(raw_numeric, "raw")
    r_map = unique_map(r_numeric, "R")

    common_keys = [key for key in r_map if key in raw_map]
    intensity_columns = [raw_map[key] for key in common_keys]
    if len(intensity_columns) < 2:
        raise ValueError(
            f"Only {len(intensity_columns)} matching intensity columns found"
        )

    id_columns = [column for column in DEFAULT_ID_COLUMNS if column in raw.columns]
    if not id_columns:
        raise ValueError("None of the default ID columns are present in the raw matrix")

    parameters_output = args.parameters_output
    if parameters_output is None:
        output = Path(args.python_output)
        parameters_output = str(
            output.with_name(output.stem + ".vsn2_parameters.csv")
        )

    _, _, summary = run_vsn2_matrix(
        input_path=args.raw_matrix,
        output_path=args.python_output,
        id_columns=id_columns,
        intensity_columns=intensity_columns,
        input_separator=raw_separator,
        output_separator=infer_separator(args.python_output, None),
        zero_is_missing=True,
        lts_quantile=args.lts_quantile,
        subsample=args.subsample,
        calib="affine",
        parameters_path=parameters_output,
        verbose=not args.quiet,
    )

    print()
    print("Generated Python VSN2 values")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("Comparison with R output")
    compare_outputs(
        python_output=args.python_output,
        r_output=args.r_output,
        details_output=args.details_output,
        differences_output=args.differences_output,
        python_separator=infer_separator(args.python_output, None),
        r_separator=r_separator,
    )

    print()
    print("The saved Python output can be compared again later with:")
    print(f'python compare.py "{args.python_output}" "{args.r_output}"')


if __name__ == "__main__":
    main()
