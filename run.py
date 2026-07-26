#python run.py --id-columns "Protein IDs" --intensity-regex "LFQ " proteinGroups.txt proteinGroups.vsn.txt
"""Run the Python VSN2 implementation on selected columns of any matrix.
Requirements:
    numpy, pandas, scipy, and vsn.py in the same directory or import path.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import vsn


def split_column_arguments(values):
    columns = []
    for value in values or []:
        columns.extend(part.strip() for part in value.split(",") if part.strip())
    return columns


def infer_separator(path, explicit_separator):
    if explicit_separator is not None:
        return "\t" if explicit_separator == r"\t" else explicit_separator
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab", ".txt"} else ","


def select_columns(all_columns, explicit_columns, regex_patterns, role):
    selected = []

    for column in explicit_columns:
        if column not in all_columns:
            raise ValueError(f"Unknown {role} column: {column!r}")
        if column not in selected:
            selected.append(column)

    for pattern in regex_patterns or []:
        compiled = re.compile(pattern)
        matches = [column for column in all_columns if compiled.search(str(column))]
        if not matches:
            raise ValueError(f"{role.capitalize()} regex matched no columns: {pattern!r}")
        for column in matches:
            if column not in selected:
                selected.append(column)

    return selected


def run_vsn2_matrix(
    input_path,
    output_path,
    id_columns,
    intensity_columns,
    id_regex=None,
    intensity_regex=None,
    input_separator=None,
    output_separator=None,
    zero_is_missing=True,
    lts_quantile=0.9,
    subsample=0,
    calib="affine",
    parameters_path=None,
    verbose=True,
):
    input_separator = infer_separator(input_path, input_separator)
    output_separator = infer_separator(output_path, output_separator)

    matrix = pd.read_csv(
        input_path,
        sep=input_separator,
        encoding="utf-8-sig",
        low_memory=False,
    )

    all_columns = list(matrix.columns)
    selected_ids = select_columns(
        all_columns,
        id_columns,
        id_regex,
        "ID",
    )
    selected_intensities = select_columns(
        all_columns,
        intensity_columns,
        intensity_regex,
        "intensity",
    )

    overlap = set(selected_ids).intersection(selected_intensities)
    if overlap:
        raise ValueError(
            "Columns cannot be both ID and intensity columns: "
            + ", ".join(sorted(overlap))
        )
    if not selected_ids:
        raise ValueError("No ID columns were selected")
    if len(selected_intensities) < 2:
        raise ValueError("VSN2 requires at least two selected intensity columns")

    intensity_matrix = (
        matrix[selected_intensities]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )
    intensity_matrix[~np.isfinite(intensity_matrix)] = np.nan
    if zero_is_missing:
        intensity_matrix[intensity_matrix == 0] = np.nan

    rows_all_missing = np.all(np.isnan(intensity_matrix), axis=1)

    fit = vsn.vsn_matrix(
        intensity_matrix,
        lts_quantile=lts_quantile,
        subsample=subsample,
        return_data=True,
        calib=calib,
        verbose=verbose,
    )

    result = matrix[selected_ids].copy()
    for index, column in enumerate(selected_intensities):
        result[column] = fit.hx[:, index]

    result.to_csv(
        output_path,
        sep=output_separator,
        index=False,
        na_rep="NA",
    )

    if parameters_path is None:
        output = Path(output_path)
        parameters_path = str(output.with_name(output.stem + ".vsn2_parameters.csv"))

    coefficients = fit.coefficients
    parameter_rows = []
    if calib == "affine":
        for stratum_index in range(coefficients.shape[0]):
            for column_index, column in enumerate(selected_intensities):
                parameter_rows.append(
                    {
                        "stratum": stratum_index + 1,
                        "intensity_column": column,
                        "offset_a": coefficients[stratum_index, column_index, 0],
                        "log_scale_b": coefficients[stratum_index, column_index, 1],
                        "scale_exp_b": np.exp(coefficients[stratum_index, column_index, 1]),
                        "hoffset": fit.hoffset[stratum_index],
                    }
                )
    else:
        parameter_rows.append(
            {
                "stratum": 1,
                "intensity_column": "ALL",
                "offset_a": coefficients[0, 0, 0],
                "log_scale_b": coefficients[0, 0, 1],
                "scale_exp_b": np.exp(coefficients[0, 0, 1]),
                "hoffset": fit.hoffset[0],
            }
        )

    pd.DataFrame(parameter_rows).to_csv(parameters_path, index=False)

    summary = {
        "rows": len(matrix),
        "id_columns": len(selected_ids),
        "intensity_columns": len(selected_intensities),
        "rows_all_missing_in_selected_intensities": int(rows_all_missing.sum()),
        "finite_input_values": int(np.isfinite(intensity_matrix).sum()),
        "finite_output_values": int(np.isfinite(fit.hx).sum()),
        "optimizer_return_code": int(fit.lbfgsb),
        "output": str(output_path),
        "parameters": str(parameters_path),
    }
    return fit, result, summary


def main():
    parser = argparse.ArgumentParser(
        description="Apply the Python VSN2 implementation to selected matrix columns."
    )
    parser.add_argument("input", help="Input CSV or TSV matrix")
    parser.add_argument("output", help="Output CSV or TSV containing IDs and VSN2 values")
    parser.add_argument(
        "--id-columns",
        action="append",
        default=[],
        metavar="COL1,COL2",
        help="ID column name(s); repeat or supply comma-separated names",
    )
    parser.add_argument(
        "--intensity-columns",
        action="append",
        default=[],
        metavar="COL1,COL2",
        help="Raw-intensity column name(s); repeat or supply comma-separated names",
    )
    parser.add_argument(
        "--id-regex",
        action="append",
        default=[],
        help="Regex selecting ID columns; may be repeated",
    )
    parser.add_argument(
        "--intensity-regex",
        action="append",
        default=[],
        help="Regex selecting raw-intensity columns; may be repeated",
    )
    parser.add_argument("--input-separator", help=r"Input separator; use \t for tab")
    parser.add_argument("--output-separator", help=r"Output separator; use \t for tab")
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="Treat zero as a measured value instead of missing",
    )
    parser.add_argument("--lts-quantile", type=float, default=0.9)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--calib", choices=["affine", "none"], default="affine")
    parser.add_argument("--parameters-output", help="Optional parameter CSV path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    fit, result, summary = run_vsn2_matrix(
        input_path=args.input,
        output_path=args.output,
        id_columns=split_column_arguments(args.id_columns),
        intensity_columns=split_column_arguments(args.intensity_columns),
        id_regex=args.id_regex,
        intensity_regex=args.intensity_regex,
        input_separator=args.input_separator,
        output_separator=args.output_separator,
        zero_is_missing=not args.keep_zero,
        lts_quantile=args.lts_quantile,
        subsample=args.subsample,
        calib=args.calib,
        parameters_path=args.parameters_output,
        verbose=not args.quiet,
    )

    print()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
