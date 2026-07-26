"""Generate VSN simulation matrices and run the local Python implementation.

Place this file beside vsn.py. It writes all outputs under vsn_simulation_results/.
"""

from pathlib import Path
import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import scipy

import vsn

OUTPUT_DIR = Path("vsn_simulation_results")
DATA_DIR = OUTPUT_DIR / "inputs"
PYTHON_DIR = OUTPUT_DIR / "python"
SEED = 20260725

SCENARIOS = [
    # name, rows, samples, missing, all-NA, log range, additive SD, multiplicative SD, outliers
    ("dense_narrow_small",       250,   4, 0.00, 0.00,  2.0,  5.0, 0.10, 0.00),
    ("dense_narrow_medium",     2000,   8, 0.00, 0.00,  2.0,  5.0, 0.10, 0.00),
    ("dense_wide_medium",       2000,   8, 0.00, 0.00, 12.0,  5.0, 0.10, 0.00),
    ("dense_wide_large",       10000,  12, 0.00, 0.00, 16.0, 10.0, 0.15, 0.00),
    ("sparse_narrow_medium",    2000,   8, 0.50, 0.10,  2.0,  5.0, 0.10, 0.00),
    ("sparse_wide_medium",      2000,   8, 0.50, 0.10, 12.0,  5.0, 0.10, 0.00),
    ("very_sparse_wide",        5000,   8, 0.75, 0.30, 12.0,  5.0, 0.10, 0.00),
    ("wide_high_additive",      2000,   8, 0.20, 0.05, 12.0, 50.0, 0.10, 0.00),
    ("wide_high_multiplicative",2000,   8, 0.20, 0.05, 12.0,  5.0, 0.40, 0.00),
    ("wide_with_outliers",      2000,   8, 0.20, 0.05, 12.0,  5.0, 0.10, 0.08),
    ("near_multiplicative_flat",2000,   8, 0.00, 0.00, 16.0,  0.01,0.01, 0.00),
    ("few_rows_wide",             60,   8, 0.20, 0.05, 12.0,  5.0, 0.10, 0.00),
    ("many_samples_wide",       2000,  24, 0.20, 0.05, 12.0,  5.0, 0.10, 0.00),
]


def simulate_matrix(rng, rows, samples, missing_fraction, all_na_fraction,
                    log_dynamic_range, additive_sd, multiplicative_sd,
                    outlier_fraction):
    log_min = np.log(1_000.0)
    feature_log_mean = rng.uniform(log_min, log_min + log_dynamic_range, rows)
    sample_log_gain = rng.normal(0.0, 0.35, samples)

    biological = rng.normal(0.0, 0.12, (rows, samples))
    multiplicative = rng.normal(0.0, multiplicative_sd, (rows, samples))
    signal = np.exp(feature_log_mean[:, None] + biological + multiplicative)
    signal *= np.exp(sample_log_gain)[None, :]
    signal += rng.normal(0.0, additive_sd, (rows, samples))
    signal[signal <= 0.0] = np.nan

    if outlier_fraction > 0.0:
        outlier_mask = rng.random((rows, samples)) < outlier_fraction
        outlier_factor = np.exp(rng.normal(0.0, 2.0, (rows, samples)))
        signal[outlier_mask] *= outlier_factor[outlier_mask]

    if missing_fraction > 0.0:
        signal[rng.random((rows, samples)) < missing_fraction] = np.nan

    all_na_count = int(round(rows * all_na_fraction))
    if all_na_count > 0:
        all_na_rows = rng.choice(rows, size=all_na_count, replace=False)
        signal[all_na_rows, :] = np.nan

    return signal


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PYTHON_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    manifest_rows = []
    run_rows = []

    for scenario in SCENARIOS:
        (name, rows, samples, missing_fraction, all_na_fraction,
         log_dynamic_range, additive_sd, multiplicative_sd,
         outlier_fraction) = scenario

        matrix = simulate_matrix(
            rng, rows, samples, missing_fraction, all_na_fraction,
            log_dynamic_range, additive_sd, multiplicative_sd,
            outlier_fraction,
        )
        sample_columns = [f"LFQ intensity sample_{i + 1:02d}" for i in range(samples)]
        input_frame = pd.DataFrame(matrix, columns=sample_columns)
        input_frame.insert(0, "feature_id", [f"{name}_{i + 1:06d}" for i in range(rows)])
        input_path = DATA_DIR / f"{name}.tsv"
        input_frame.to_csv(input_path, sep="\t", index=False, na_rep="NA", float_format="%.17g")

        start = pd.Timestamp.now(tz="UTC")
        error = ""
        optimizer_code = -999
        finite_output = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = vsn.vsn_matrix(
                    np.asarray(matrix, dtype=np.float64),
                    lts_quantile=0.9,
                    subsample=0,
                    calib="affine",
                    min_data_points_per_stratum=0,
                    verbose=False,
                )
            normalized = np.asarray(fit.hx, dtype=np.float64)
            optimizer_code = int(fit.lbfgsb)
            finite_output = int(np.isfinite(normalized).sum())
            output_frame = pd.DataFrame(normalized, columns=[f"vsn_{c}" for c in sample_columns])
            output_frame.insert(0, "feature_id", input_frame["feature_id"])
            output_frame.to_csv(
                PYTHON_DIR / f"{name}.python.tsv",
                sep="\t", index=False, na_rep="NA", float_format="%.17g",
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        elapsed = (pd.Timestamp.now(tz="UTC") - start).total_seconds()
        counts = np.sum(np.isfinite(matrix), axis=1)
        finite_values = matrix[np.isfinite(matrix)]
        manifest_rows.append({
            "scenario": name,
            "input_file": input_path.as_posix(),
            "rows": rows,
            "samples": samples,
            "requested_missing_fraction": missing_fraction,
            "requested_all_na_fraction": all_na_fraction,
            "log_dynamic_range": log_dynamic_range,
            "additive_sd": additive_sd,
            "multiplicative_sd": multiplicative_sd,
            "outlier_fraction": outlier_fraction,
            "lts_quantile": 0.9,
            "calib": "affine",
        })
        run_rows.append({
            "scenario": name,
            "rows": rows,
            "samples": samples,
            "finite_input_values": int(np.isfinite(matrix).sum()),
            "actual_missing_fraction": float(np.mean(~np.isfinite(matrix))),
            "all_na_rows": int(np.sum(counts == 0)),
            "rows_with_one_value": int(np.sum(counts == 1)),
            "minimum_finite_value": float(np.min(finite_values)) if finite_values.size else np.nan,
            "maximum_finite_value": float(np.max(finite_values)) if finite_values.size else np.nan,
            "optimizer_return_code": optimizer_code,
            "finite_output_values": finite_output,
            "elapsed_seconds": elapsed,
            "error": error,
        })
        print(name, "optimizer", optimizer_code, "error", error or "none")

    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "manifest.tsv", sep="\t", index=False)
    pd.DataFrame(run_rows).to_csv(OUTPUT_DIR / "python_run_summary.tsv", sep="\t", index=False)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "seed": SEED,
    }
    (OUTPUT_DIR / "python_environment.json").write_text(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
