"""Compare native R and Python VSN simulation outputs.

This version reports missing or failed R/Python outputs cleanly instead of
raising KeyError when no scenario produced a comparable output pair.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def read_tsv(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def clean_error(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="vsn_simulation_results")
    args = parser.parse_args()
    root = Path(args.root)

    manifest_path = root / "manifest.tsv"
    python_summary_path = root / "python_run_summary.tsv"
    r_summary_path = root / "r_run_summary.tsv"

    for required_path in (manifest_path, python_summary_path, r_summary_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    manifest = read_tsv(manifest_path)
    python_summary = read_tsv(python_summary_path)
    r_summary = read_tsv(r_summary_path)

    python_summary_by_scenario = python_summary.set_index("scenario", drop=False)
    r_summary_by_scenario = r_summary.set_index("scenario", drop=False)

    scenario_rows = []
    sample_rows = []

    for scenario in manifest["scenario"].astype(str):
        python_path = root / "python" / f"{scenario}.python.tsv"
        r_path = root / "r" / f"{scenario}.r.tsv"

        python_error = ""
        r_error = ""
        if scenario in python_summary_by_scenario.index:
            python_error = clean_error(
                python_summary_by_scenario.loc[scenario, "error"]
            )
        if scenario in r_summary_by_scenario.index:
            r_error = clean_error(r_summary_by_scenario.loc[scenario, "error"])

        base_result = {
            "scenario": scenario,
            "python_file_exists": python_path.exists(),
            "r_file_exists": r_path.exists(),
            "python_error": python_error,
            "r_error": r_error,
            "finite_pairs": 0,
            "missing_mismatches": np.nan,
            "mean_python_minus_r": np.nan,
            "median_python_minus_r": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "p95_absolute_error": np.nan,
            "p99_absolute_error": np.nan,
            "max_absolute_error": np.nan,
            "fraction_absolute_error_gt_1e_8": np.nan,
            "fraction_absolute_error_gt_1e_6": np.nan,
            "fraction_absolute_error_gt_1e_4": np.nan,
            "all_absolute_differences_le_1e_8": False,
            "all_absolute_differences_le_1e_6": False,
        }

        if not python_path.exists() or not r_path.exists():
            missing_parts = []
            if not python_path.exists():
                missing_parts.append("Python output missing")
            if not r_path.exists():
                missing_parts.append("R output missing")
            if python_error:
                missing_parts.append(f"Python error: {python_error}")
            if r_error:
                missing_parts.append(f"R error: {r_error}")
            base_result["comparison_status"] = "; ".join(missing_parts)
            scenario_rows.append(base_result)
            continue

        try:
            python_frame = read_tsv(python_path)
            r_frame = read_tsv(r_path)

            if "feature_id" not in python_frame.columns:
                raise ValueError(f"feature_id missing from {python_path}")
            if "feature_id" not in r_frame.columns:
                raise ValueError(f"feature_id missing from {r_path}")

            python_value_columns = [
                column for column in python_frame.columns
                if column.startswith("vsn_")
            ]
            r_value_columns = [
                column for column in r_frame.columns
                if column.startswith("vsn_")
            ]
            shared_value_columns = [
                column for column in r_value_columns
                if column in python_value_columns
            ]

            if not shared_value_columns:
                raise ValueError(
                    "No shared vsn_ columns. "
                    f"R columns: {r_value_columns}; "
                    f"Python columns: {python_value_columns}"
                )

            merged = r_frame[["feature_id"] + shared_value_columns].merge(
                python_frame[["feature_id"] + shared_value_columns],
                on="feature_id",
                suffixes=("_r", "_python"),
                validate="one_to_one",
            )

            all_differences = []
            missing_mismatch_total = 0

            for value_column in shared_value_columns:
                r_column = f"{value_column}_r"
                python_column = f"{value_column}_python"
                r_values = pd.to_numeric(
                    merged[r_column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                python_values = pd.to_numeric(
                    merged[python_column], errors="coerce"
                ).to_numpy(dtype=np.float64)

                finite_pairs = np.isfinite(r_values) & np.isfinite(python_values)
                missing_mismatches = int(
                    np.sum(np.isfinite(r_values) != np.isfinite(python_values))
                )
                differences = python_values[finite_pairs] - r_values[finite_pairs]
                missing_mismatch_total += missing_mismatches
                if differences.size:
                    all_differences.append(differences)

                sample_rows.append({
                    "scenario": scenario,
                    "sample": value_column.removeprefix("vsn_"),
                    "finite_pairs": int(finite_pairs.sum()),
                    "missing_mismatches": missing_mismatches,
                    "mean_python_minus_r": (
                        float(np.mean(differences)) if differences.size else np.nan
                    ),
                    "median_python_minus_r": (
                        float(np.median(differences)) if differences.size else np.nan
                    ),
                    "rmse": (
                        float(np.sqrt(np.mean(differences ** 2)))
                        if differences.size else np.nan
                    ),
                    "mae": (
                        float(np.mean(np.abs(differences)))
                        if differences.size else np.nan
                    ),
                    "max_absolute_error": (
                        float(np.max(np.abs(differences)))
                        if differences.size else np.nan
                    ),
                    "pearson_r": safe_corr(
                        r_values[finite_pairs], python_values[finite_pairs]
                    ),
                    "allclose_rtol_1e_8_atol_1e_8": bool(
                        np.allclose(
                            r_values,
                            python_values,
                            rtol=1e-8,
                            atol=1e-8,
                            equal_nan=True,
                        )
                    ),
                    "allclose_rtol_1e_6_atol_1e_6": bool(
                        np.allclose(
                            r_values,
                            python_values,
                            rtol=1e-6,
                            atol=1e-6,
                            equal_nan=True,
                        )
                    ),
                })

            differences = (
                np.concatenate(all_differences)
                if all_differences else np.array([], dtype=np.float64)
            )
            if not differences.size:
                raise ValueError("No finite R/Python value pairs were found")

            absolute_differences = np.abs(differences)
            base_result.update({
                "comparison_status": "ok",
                "finite_pairs": int(differences.size),
                "missing_mismatches": missing_mismatch_total,
                "mean_python_minus_r": float(np.mean(differences)),
                "median_python_minus_r": float(np.median(differences)),
                "rmse": float(np.sqrt(np.mean(differences ** 2))),
                "mae": float(np.mean(absolute_differences)),
                "p95_absolute_error": float(
                    np.quantile(absolute_differences, 0.95)
                ),
                "p99_absolute_error": float(
                    np.quantile(absolute_differences, 0.99)
                ),
                "max_absolute_error": float(np.max(absolute_differences)),
                "fraction_absolute_error_gt_1e_8": float(
                    np.mean(absolute_differences > 1e-8)
                ),
                "fraction_absolute_error_gt_1e_6": float(
                    np.mean(absolute_differences > 1e-6)
                ),
                "fraction_absolute_error_gt_1e_4": float(
                    np.mean(absolute_differences > 1e-4)
                ),
                "all_absolute_differences_le_1e_8": bool(
                    np.all(absolute_differences <= 1e-8)
                ),
                "all_absolute_differences_le_1e_6": bool(
                    np.all(absolute_differences <= 1e-6)
                ),
            })
        except Exception as exc:
            base_result["comparison_status"] = (
                f"comparison error: {type(exc).__name__}: {exc}"
            )

        scenario_rows.append(base_result)

    scenario_results = pd.DataFrame(scenario_rows)
    scenario_table = manifest.merge(scenario_results, on="scenario", how="left")

    python_status = python_summary[[
        "scenario", "optimizer_return_code", "elapsed_seconds"
    ]].rename(columns={
        "optimizer_return_code": "python_optimizer_code",
        "elapsed_seconds": "python_elapsed_seconds",
    })
    r_status = r_summary[[
        "scenario", "optimizer_return_code", "elapsed_seconds"
    ]].rename(columns={
        "optimizer_return_code": "r_optimizer_code",
        "elapsed_seconds": "r_elapsed_seconds",
    })
    scenario_table = scenario_table.merge(
        python_status, on="scenario", how="left"
    ).merge(r_status, on="scenario", how="left")

    summary_path = root / "comparison_summary.tsv"
    by_sample_path = root / "comparison_by_sample.tsv"
    ranked_path = root / "comparison_ranked_by_max_error.tsv"

    scenario_table.to_csv(
        summary_path, sep="\t", index=False, float_format="%.17g"
    )
    pd.DataFrame(sample_rows).to_csv(
        by_sample_path, sep="\t", index=False, float_format="%.17g"
    )

    successful = scenario_table[
        scenario_table["comparison_status"].eq("ok")
    ].copy()
    if not successful.empty:
        ranked = successful.sort_values(
            "max_absolute_error", ascending=False, na_position="last"
        )
        ranked.to_csv(
            ranked_path, sep="\t", index=False, float_format="%.17g"
        )
        print(ranked[[
            "scenario", "rows", "samples",
            "requested_missing_fraction", "log_dynamic_range",
            "finite_pairs", "rmse", "max_absolute_error",
            "missing_mismatches", "python_optimizer_code",
            "r_optimizer_code",
        ]].to_string(index=False))
    else:
        scenario_table.to_csv(
            ranked_path, sep="\t", index=False, float_format="%.17g"
        )
        print("No successful R/Python comparison pairs were found.")
        print("\nPer-scenario status:")
        print(scenario_table[[
            "scenario", "comparison_status", "python_file_exists",
            "r_file_exists", "python_error", "r_error",
        ]].to_string(index=False))
        print("\nInspect r_run_summary.tsv and python_run_summary.tsv.")

    print("\nShare these files:")
    for path in (
        summary_path,
        by_sample_path,
        ranked_path,
        root / "python_run_summary.tsv",
        root / "r_run_summary.tsv",
        root / "python_environment.json",
        root / "r_session_info.txt",
    ):
        print(path)


if __name__ == "__main__":
    main()
