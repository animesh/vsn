#python compare.py "python_vsn_matched_samples.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv"
#python compare.py "report.pg.vsn2.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv"
"""Compare VSN values from a Python TSV/CSV with an R CorTestBH CSV.
The script compares only intensity columns present in both files. It ignores
ID/statistics columns and handles R's make.names formatting plus the appended
"..value" suffix used in the supplied CorTestBH output.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


R_NON_INTENSITY_COLUMNS = {
    "Uniprot",
    "Gene",
    "PValueMinusLog10",
    "CorrectedPValueBH",
    "CorTestPval",
    "Cor",
    "Row",
}


def infer_separator(path, explicit=None):
    if explicit is not None:
        return "\t" if explicit == r"\t" else explicit
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab", ".txt"} else ","


def canonical_sample(column):
    """Convert Python/raw and R-formatted sample headers to one key."""
    name = str(column).strip().replace("\\", "/").rsplit("/", 1)[-1]

    # R check.names=TRUE prefixes names beginning with a number with X.
    if re.match(r"^X\d", name):
        name = name[1:]

    # R workflow appends the correlation covariate as '..77', '..100', etc.
    name = re.sub(r"\.\.[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$", "", name)

    # Remove the experiment-specific prefix if present.
    name = re.sub(r"^260518_Sonali_", "", name)

    # Remove DIA-NN suffix.
    name = re.sub(r"\.d$", "", name, flags=re.IGNORECASE)

    # R make.names converts '-' to '.', while underscores remain underscores.
    name = name.replace(".", "-")
    return name


def numeric_candidate_columns(frame, excluded):
    candidates = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            candidates.append(column)
    return candidates


def build_unique_map(columns, label):
    mapping = {}
    duplicates = {}
    for column in columns:
        key = canonical_sample(column)
        if key in mapping:
            duplicates.setdefault(key, [mapping[key]]).append(column)
        else:
            mapping[key] = column
    if duplicates:
        details = "; ".join(f"{key}: {values}" for key, values in duplicates.items())
        raise ValueError(f"Duplicate canonical {label} sample names: {details}")
    return mapping


def compare_outputs(
    python_output,
    r_output,
    details_output,
    differences_output=None,
    python_separator=None,
    r_separator=None,
):
    python_frame = pd.read_csv(
        python_output,
        sep=infer_separator(python_output, python_separator),
        encoding="utf-8-sig",
        low_memory=False,
    )
    r_frame = pd.read_csv(
        r_output,
        sep=infer_separator(r_output, r_separator),
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(python_frame) != len(r_frame):
        raise ValueError(
            f"Row-count mismatch: Python has {len(python_frame)} rows, "
            f"R has {len(r_frame)} rows"
        )

    python_columns = numeric_candidate_columns(python_frame, excluded=set())
    r_columns = numeric_candidate_columns(r_frame, excluded=R_NON_INTENSITY_COLUMNS)

    python_map = build_unique_map(python_columns, "Python")
    r_map = build_unique_map(r_columns, "R")

    common_keys = [key for key in r_map if key in python_map]
    if not common_keys:
        raise ValueError(
            "No matching intensity columns were found. Check the headers and separators."
        )

    detail_rows = []
    all_differences = []
    difference_frame = pd.DataFrame(index=np.arange(len(python_frame)))

    for key in common_keys:
        python_column = python_map[key]
        r_column = r_map[key]
        python_values = pd.to_numeric(
            python_frame[python_column], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        r_values = pd.to_numeric(
            r_frame[r_column], errors="coerce"
        ).to_numpy(dtype=float, copy=True)

        valid = np.isfinite(python_values) & np.isfinite(r_values)
        if not np.any(valid):
            continue

        differences = python_values[valid] - r_values[valid]
        all_differences.append(differences)

        correlation = np.nan
        if valid.sum() >= 2:
            correlation = float(np.corrcoef(python_values[valid], r_values[valid])[0, 1])

        detail_rows.append(
            {
                "sample_key": key,
                "python_column": python_column,
                "r_column": r_column,
                "compared_values": int(valid.sum()),
                "python_missing": int((~np.isfinite(python_values)).sum()),
                "r_missing": int((~np.isfinite(r_values)).sum()),
                "pearson_correlation": correlation,
                "r_squared": correlation**2 if np.isfinite(correlation) else np.nan,
                "rmse": float(np.sqrt(np.mean(differences**2))),
                "mae": float(np.mean(np.abs(differences))),
                "mean_difference_python_minus_r": float(np.mean(differences)),
                "maximum_absolute_error": float(np.max(np.abs(differences))),
            }
        )

        if differences_output is not None:
            full_difference = np.full(len(python_frame), np.nan)
            full_difference[valid] = differences
            difference_frame[key] = full_difference

    if not all_differences:
        raise ValueError("Matching columns were found, but no finite value pairs were available")

    details = pd.DataFrame(detail_rows)
    details.to_csv(details_output, index=False)

    if differences_output is not None:
        difference_frame.to_csv(differences_output, index=False, na_rep="NA")

    combined = np.concatenate(all_differences)
    unmatched_python = [column for key, column in python_map.items() if key not in r_map]
    unmatched_r = [column for key, column in r_map.items() if key not in python_map]

    print(f"Rows in each file: {len(python_frame)}")
    print(f"Matched intensity columns: {len(details)}")
    print(f"Compared finite values: {combined.size}")
    print(f"Overall RMSE: {np.sqrt(np.mean(combined**2)):.12g}")
    print(f"Overall MAE: {np.mean(np.abs(combined)):.12g}")
    print(f"Overall mean difference (Python - R): {np.mean(combined):.12g}")
    print(f"Overall maximum absolute error: {np.max(np.abs(combined)):.12g}")
    print(f"Unmatched Python numeric columns: {len(unmatched_python)}")
    print(f"Unmatched R intensity columns: {len(unmatched_r)}")
    print(f"Per-sample details: {details_output}")
    if differences_output is not None:
        print(f"Row-level differences: {differences_output}")

    if unmatched_r:
        print("R columns not matched:")
        for column in unmatched_r:
            print(f"  {column}")

    return details, combined


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python VSN output with R VSN values."
    )
    parser.add_argument("python_output", help="Python VSN CSV/TSV")
    parser.add_argument("r_output", help="R CorTestBH CSV/TSV")
    parser.add_argument(
        "--details-output",
        default="vsn_output_comparison_by_sample.csv",
        help="Per-sample metrics CSV",
    )
    parser.add_argument(
        "--differences-output",
        help="Optional row-by-row Python-minus-R differences CSV",
    )
    parser.add_argument("--python-separator", help=r"Use \t for tab")
    parser.add_argument("--r-separator", help=r"Use \t for tab")
    args = parser.parse_args()

    compare_outputs(
        python_output=args.python_output,
        r_output=args.r_output,
        details_output=args.details_output,
        differences_output=args.differences_output,
        python_separator=args.python_separator,
        r_separator=args.r_separator,
    )


if __name__ == "__main__":
    main()
