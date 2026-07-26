"""Compare VSN outputs produced by run.py and run.r.

Expected workflows
------------------
R:
    Rscript run.r proteinGroups.txt LFQ
    -> proteinGroups.txt.LFQ.vsn.txt

Python:
    python run.py --id-columns "Protein IDs" --intensity-regex "LFQ " \
        proteinGroups.txt proteinGroups.vsn.txt

The R file contains original columns plus columns named ``vsn_<R header>``.
The Python runner output contains the selected ID columns plus normalized values
under their original Python/MaxQuant headers. This script selects only those two
normalized column sets, canonicalizes R ``make.names`` punctuation, aligns rows
by protein ID when possible, and writes overall, per-sample, and optional
row-level comparison tables.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def infer_separator(path, explicit=None):
    if explicit is not None:
        return "\t" if explicit == r"\t" else explicit
    return "\t" if Path(path).suffix.lower() in {".tsv", ".tab", ".txt"} else ","


def read_table(path, separator=None):
    return pd.read_csv(
        path,
        sep=infer_separator(path, separator),
        encoding="utf-8-sig",
        low_memory=False,
    )


def canonical_text(value):
    """Normalize an identifier/header for punctuation-insensitive matching."""
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def canonical_sample(column):
    """Map Python and R VSN headers to the same sample key."""
    name = str(column).strip().replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"^vsn[_. -]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^lfq[_. -]*intensity[_. -]*", "", name, flags=re.IGNORECASE)
    name = name.strip("._ -")
    # R read.delim(check.names=TRUE) changes spaces/hyphens to dots.
    # Normalize every punctuation run so names from both files agree.
    return canonical_text(name)


def choose_id_column(frame, requested, candidates, label):
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"{label} ID column not found: {requested!r}")
        return requested
    canonical_columns = {canonical_text(column): column for column in frame.columns}
    for candidate in candidates:
        matched = canonical_columns.get(canonical_text(candidate))
        if matched is not None:
            return matched
    return None


def select_python_vsn_columns(frame, prefix):
    selected = [column for column in frame.columns if str(column).startswith(prefix)]
    if not selected:
        raise ValueError(
            f"No Python normalized columns start with {prefix!r}. "
            "Pass --python-prefix matching the --intensity-regex/column names used by run.py."
        )
    return selected


def select_r_vsn_columns(frame, prefix):
    selected = [column for column in frame.columns if str(column).startswith(prefix)]
    if not selected:
        raise ValueError(
            f"No R normalized columns start with {prefix!r}. "
            "Pass --r-prefix matching columns written by run.r."
        )
    return selected


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


def align_frames(python_frame, r_frame, python_id, r_id):
    if python_id is None or r_id is None:
        if len(python_frame) != len(r_frame):
            raise ValueError(
                "No usable ID columns found and row counts differ: "
                f"Python={len(python_frame)}, R={len(r_frame)}"
            )
        aligned_python = python_frame.reset_index(drop=True)
        aligned_r = r_frame.reset_index(drop=True)
        identifiers = pd.Series(np.arange(len(aligned_python)), name="row_index")
        return aligned_python, aligned_r, identifiers, "row order"

    python_ids = python_frame[python_id].astype("string")
    r_ids = r_frame[r_id].astype("string")
    if python_ids.isna().any() or r_ids.isna().any():
        raise ValueError("ID columns contain missing values; cannot safely align rows.")
    if python_ids.duplicated().any():
        examples = python_ids[python_ids.duplicated(keep=False)].head().tolist()
        raise ValueError(f"Python ID column is not unique; examples: {examples}")
    if r_ids.duplicated().any():
        examples = r_ids[r_ids.duplicated(keep=False)].head().tolist()
        raise ValueError(f"R ID column is not unique; examples: {examples}")

    python_indexed = python_frame.copy()
    r_indexed = r_frame.copy()
    python_indexed["__comparison_id__"] = python_ids
    r_indexed["__comparison_id__"] = r_ids
    common_ids = python_indexed["__comparison_id__"].isin(
        r_indexed["__comparison_id__"]
    )
    python_aligned = python_indexed.loc[common_ids].set_index("__comparison_id__")
    r_aligned = r_indexed.set_index("__comparison_id__").reindex(python_aligned.index)
    if len(python_aligned) == 0:
        raise ValueError("The Python and R ID columns have no values in common.")
    identifiers = pd.Series(python_aligned.index, name=python_id).reset_index(drop=True)
    return (
        python_aligned.reset_index(drop=True),
        r_aligned.reset_index(drop=True),
        identifiers,
        f"IDs: Python {python_id!r}, R {r_id!r}",
    )


def compare_outputs(
    python_output,
    r_output,
    details_output="vsn_comparison_by_sample.tsv",
    summary_output="vsn_comparison_summary.tsv",
    differences_output="vsn_differences.tsv",
    python_separator=None,
    r_separator=None,
    python_prefix="LFQ intensity ",
    r_prefix="vsn_LFQ.intensity.",
    python_id=None,
    r_id=None,
):
    python_frame = read_table(python_output, python_separator)
    r_frame = read_table(r_output, r_separator)

    python_id = choose_id_column(
        python_frame,
        python_id,
        ["Protein IDs", "Protein.IDs", "Protein.Group", "Protein.Group.IDs"],
        "Python",
    )
    r_id = choose_id_column(
        r_frame,
        r_id,
        ["Protein.IDs", "Protein IDs", "Protein.Group", "Protein.Group.IDs"],
        "R",
    )
    python_frame, r_frame, identifiers, alignment = align_frames(
        python_frame, r_frame, python_id, r_id
    )

    python_columns = select_python_vsn_columns(python_frame, python_prefix)
    r_columns = select_r_vsn_columns(r_frame, r_prefix)
    python_map = build_unique_map(python_columns, "Python")
    r_map = build_unique_map(r_columns, "R")
    common_keys = [key for key in python_map if key in r_map]
    if not common_keys:
        raise ValueError(
            "No normalized sample columns matched after canonicalization.\n"
            f"Python candidates: {python_columns}\nR candidates: {r_columns}"
        )

    details = []
    combined_differences = []
    difference_frame = pd.DataFrame({identifiers.name: identifiers})
    missing_mismatch_total = 0

    for key in common_keys:
        python_column = python_map[key]
        r_column = r_map[key]
        python_values = pd.to_numeric(
            python_frame[python_column], errors="coerce"
        ).to_numpy(dtype=np.float64)
        r_values = pd.to_numeric(
            r_frame[r_column], errors="coerce"
        ).to_numpy(dtype=np.float64)

        python_finite = np.isfinite(python_values)
        r_finite = np.isfinite(r_values)
        valid = python_finite & r_finite
        missing_mismatches = int(np.sum(python_finite != r_finite))
        missing_mismatch_total += missing_mismatches
        if not valid.any():
            continue

        differences = python_values[valid] - r_values[valid]
        absolute_differences = np.abs(differences)
        combined_differences.append(differences)
        correlation = (
            float(np.corrcoef(python_values[valid], r_values[valid])[0, 1])
            if valid.sum() >= 2
            else np.nan
        )
        details.append(
            {
                "sample_key": key,
                "python_column": python_column,
                "r_column": r_column,
                "compared_values": int(valid.sum()),
                "python_missing": int((~python_finite).sum()),
                "r_missing": int((~r_finite).sum()),
                "missing_mismatches": missing_mismatches,
                "pearson_correlation": correlation,
                "r_squared": correlation**2 if np.isfinite(correlation) else np.nan,
                "rmse": float(np.sqrt(np.mean(differences**2))),
                "mae": float(np.mean(absolute_differences)),
                "mean_difference_python_minus_r": float(np.mean(differences)),
                "median_difference_python_minus_r": float(np.median(differences)),
                "maximum_absolute_error": float(np.max(absolute_differences)),
                "allclose_rtol_1e_8_atol_1e_8": bool(
                    np.allclose(
                        python_values,
                        r_values,
                        rtol=1e-8,
                        atol=1e-8,
                        equal_nan=True,
                    )
                ),
                "allclose_rtol_1e_7_atol_1e_6": bool(
                    np.allclose(
                        python_values,
                        r_values,
                        rtol=1e-7,
                        atol=1e-6,
                        equal_nan=True,
                    )
                ),
            }
        )
        if differences_output:
            full_difference = np.full(len(python_frame), np.nan)
            full_difference[valid] = differences
            difference_frame[f"python_minus_r__{python_column}"] = full_difference

    if not combined_differences:
        raise ValueError("Matching columns were found, but no finite pairs were available.")

    details_frame = pd.DataFrame(details)
    combined = np.concatenate(combined_differences)
    absolute_combined = np.abs(combined)
    unmatched_python = [python_map[key] for key in python_map if key not in r_map]
    unmatched_r = [r_map[key] for key in r_map if key not in python_map]

    summary_frame = pd.DataFrame(
        [
            {
                "python_file": str(python_output),
                "r_file": str(r_output),
                "row_alignment": alignment,
                "aligned_rows": len(python_frame),
                "matched_intensity_columns": len(details_frame),
                "compared_finite_values": int(combined.size),
                "missing_mismatches": missing_mismatch_total,
                "overall_rmse": float(np.sqrt(np.mean(combined**2))),
                "overall_mae": float(np.mean(absolute_combined)),
                "overall_mean_difference_python_minus_r": float(np.mean(combined)),
                "overall_median_difference_python_minus_r": float(np.median(combined)),
                "overall_maximum_absolute_error": float(np.max(absolute_combined)),
                "all_absolute_differences_le_1e_8": bool(
                    np.all(absolute_combined <= 1e-8)
                ),
                "all_absolute_differences_le_1e_6": bool(
                    np.all(absolute_combined <= 1e-6)
                ),
                "unmatched_python_columns": len(unmatched_python),
                "unmatched_r_columns": len(unmatched_r),
            }
        ]
    )

    details_frame.to_csv(details_output, sep="\t", index=False, float_format="%.17g")
    summary_frame.to_csv(summary_output, sep="\t", index=False, float_format="%.17g")
    if differences_output:
        difference_frame.to_csv(
            differences_output, sep="\t", index=False, na_rep="NA", float_format="%.17g"
        )

    summary = summary_frame.iloc[0]
    print(f"Row alignment: {alignment}")
    print(f"Aligned rows: {summary['aligned_rows']}")
    print(f"Matched normalized columns: {summary['matched_intensity_columns']}")
    print(f"Compared finite values: {summary['compared_finite_values']}")
    print(f"Missing-value mismatches: {summary['missing_mismatches']}")
    print(f"Overall RMSE: {summary['overall_rmse']:.12g}")
    print(f"Overall MAE: {summary['overall_mae']:.12g}")
    print(
        "Overall mean difference (Python - R): "
        f"{summary['overall_mean_difference_python_minus_r']:.12g}"
    )
    print(
        "Overall maximum absolute error: "
        f"{summary['overall_maximum_absolute_error']:.12g}"
    )
    print(f"Summary: {summary_output}")
    print(f"Per-sample details: {details_output}")
    if differences_output:
        print(f"Row-level differences: {differences_output}")
    if unmatched_python:
        print("Unmatched Python normalized columns:")
        for column in unmatched_python:
            print(f"  {column}")
    if unmatched_r:
        print("Unmatched R normalized columns:")
        for column in unmatched_r:
            print(f"  {column}")

    return summary_frame, details_frame, combined


def main():
    parser = argparse.ArgumentParser(
        description="Compare VSN outputs produced by run.py and run.r."
    )
    parser.add_argument(
        "python_output",
        nargs="?",
        default="proteinGroups.vsn.txt",
        help="Output file written by run.py (default: proteinGroups.vsn.txt)",
    )
    parser.add_argument(
        "r_output",
        nargs="?",
        default="proteinGroups.txt.LFQ.vsn.txt",
        help=(
            "Output file written by run.r "
            "(default: proteinGroups.txt.LFQ.vsn.txt)"
        ),
    )
    parser.add_argument(
        "--details-output",
        default="vsn_comparison_by_sample.tsv",
        help="Per-sample metrics TSV",
    )
    parser.add_argument(
        "--summary-output",
        default="vsn_comparison_summary.tsv",
        help="Overall metrics TSV",
    )
    parser.add_argument(
        "--differences-output",
        default="vsn_differences.tsv",
        help=(
            "Row-level Python-minus-R differences TSV "
            "(default: vsn_differences.tsv)"
        ),
    )
    parser.add_argument("--python-separator", help=r"Explicit separator; use \t for tab")
    parser.add_argument("--r-separator", help=r"Explicit separator; use \t for tab")
    parser.add_argument(
        "--python-prefix",
        default="LFQ intensity ",
        help="Prefix of normalized columns written by run.py",
    )
    parser.add_argument(
        "--r-prefix",
        default="vsn_LFQ.intensity.",
        help="Prefix of normalized columns written by run.r",
    )
    parser.add_argument("--python-id", help="Python ID column; auto-detected by default")
    parser.add_argument("--r-id", help="R ID column; auto-detected by default")
    args = parser.parse_args()

    compare_outputs(
        python_output=args.python_output,
        r_output=args.r_output,
        details_output=args.details_output,
        summary_output=args.summary_output,
        differences_output=args.differences_output,
        python_separator=args.python_separator,
        r_separator=args.r_separator,
        python_prefix=args.python_prefix,
        r_prefix=args.r_prefix,
        python_id=args.python_id,
        r_id=args.r_id,
    )


if __name__ == "__main__":
    main()
