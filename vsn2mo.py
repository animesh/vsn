# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "numpy>=2.5.1",
#     "pandas>=3.0.3",
#     "scipy>=1.18.0",
#     "matplotlib>=3.10.0",
# ]
# ///
#local view: uv run marimo run vsn2mo.py
#server edit: animeshs@ubuntu:~/scripts$ uv run marimo edit vsn2mo.py --host 0.0.0.0 --port 2718

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="VSN Normalization Dashboard")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import io
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from scipy.stats import rankdata
    from scipy.stats import linregress
    import warnings
    from datetime import datetime
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import Optional

    return Optional, Path, dataclass, datetime, field, io, linregress, minimize, mo, np, pd, plt, rankdata, warnings


@app.cell(hide_code=True)
def _(minimize, np, rankdata, warnings):
    DEFAULT_OPTIMPAR = {
        "factr": 5e7,
        "pgtol": 2e-4,
        "maxit": 60000,
        "trace": 0,
        "cvg_niter": 7,
        "cvg_eps": 0.0,
    }
    OPTIMPAR_NAMES = set(DEFAULT_OPTIMPAR)

    def _unpack_parameters(_p, _nc, _calib):
        if _calib == "affine":
            return _p[:_nc], _p[_nc:]
        return np.full(_nc, _p[0]), np.full(_nc, _p[1])

    def _negloglik_and_grad(_p, _x, _calib):
        """Profile VSN negative log-likelihood and analytical gradient."""
        _nr, _nc = _x.shape
        if _calib == "none":
            _flat = _x.flatten(order="F")
            _valid_flat = np.isfinite(_flat)
            _nt_none = int(_valid_flat.sum())
            _a_none = float(_p[0])
            _logb_none = float(_p[1])
            _b_none = float(np.exp(_logb_none))
            _u_none = np.full_like(_flat, np.nan)
            _h_none = np.full_like(_flat, np.nan)
            _ma_none = np.full_like(_flat, np.nan)
            _u_none[_valid_flat] = _b_none * _flat[_valid_flat] + _a_none
            _h_none[_valid_flat] = np.arcsinh(_u_none[_valid_flat])
            _ma_none[_valid_flat] = 1.0 / np.sqrt(1.0 + _u_none[_valid_flat] ** 2)
            _h_none_matrix = _h_none.reshape((_nr, _nc), order="F")
            _mu_none = np.nanmean(_h_none_matrix, axis=1)
            _resid_none = (_h_none_matrix - _mu_none[:, np.newaxis]).flatten(order="F")
            _ssq_none = float(np.nansum(_resid_none**2))
            if _nt_none == 0 or not np.isfinite(_ssq_none) or _ssq_none <= 0:
                return 1e15, np.zeros_like(_p)
            _sigsq_none = _ssq_none / _nt_none
            _nll_none = (
                (_nt_none / 2.0) * np.log(2.0 * np.pi * _sigsq_none)
                + (_nt_none / 2.0)
                + 0.5 * float(np.sum(np.log1p(_u_none[_valid_flat] ** 2)))
                - _nt_none * _logb_none
            )
            _z_none = (
                _resid_none[_valid_flat] / _sigsq_none
                + _ma_none[_valid_flat] * _u_none[_valid_flat]
            )
            _grad_a_none = float(np.sum(_z_none * _ma_none[_valid_flat]))
            _grad_b_none = float(
                _b_none * (
                    np.sum(_z_none * _ma_none[_valid_flat] * _flat[_valid_flat])
                    - _nt_none / _b_none
                )
            )
            return float(_nll_none), np.array([_grad_a_none, _grad_b_none])

        if _calib == "affine":
            _a = _p[:_nc]
            _logb = _p[_nc:]
        else:
            _a = np.full(_nc, _p[0])
            _logb = np.full(_nc, _p[1])
        _b = np.exp(_logb)

        _h = np.full((_nr, _nc), np.nan)
        _ly = np.full((_nr, _nc), np.nan)
        _ma = np.full((_nr, _nc), np.nan)
        _jac1 = 0.0
        _jac2 = 0.0
        _nj_arr = np.zeros(_nc, dtype=int)

        for _j in range(_nc):
            _col = _x[:, _j]
            _valid = np.isfinite(_col)
            _nj = int(_valid.sum())
            _nj_arr[_j] = _nj
            if _nj == 0:
                continue
            _u = _b[_j] * _col[_valid] + _a[_j]
            _h[_valid, _j] = np.arcsinh(_u)
            _ly[_valid, _j] = _u
            _ma[_valid, _j] = 1.0 / np.sqrt(1.0 + _u**2)
            _jac1 += float(np.sum(np.log1p(_u**2)))
            _jac2 += _nj * _logb[_j]

        _mu = np.nanmean(_h, axis=1)
        _resid = _h - _mu[:, np.newaxis]
        _ssq = float(np.nansum(_resid**2))
        _nt = int(np.sum(np.isfinite(_x)))
        if _nt == 0 or not np.isfinite(_ssq) or _ssq <= 0:
            return 1e15, np.zeros_like(_p)
        _sigsq = _ssq / _nt
        _nll = (
            (_nt / 2.0) * np.log(2.0 * np.pi * _sigsq)
            + (_nt / 2.0)
            + 0.5 * _jac1
            - _jac2
        )

        _rfac = 1.0 / _sigsq
        if _calib == "affine":
            _grad_a = np.zeros(_nc)
            _grad_logb = np.zeros(_nc)
        else:
            _grad_a = 0.0
            _grad_logb = 0.0

        for _j in range(_nc):
            _valid = np.isfinite(_x[:, _j]) & np.isfinite(_resid[:, _j])
            if not _valid.any():
                continue
            _z = (
                _resid[_valid, _j] * _rfac
                + _ma[_valid, _j] * _ly[_valid, _j]
            )
            _sa = float(np.sum(_z * _ma[_valid, _j]))
            _sb = float(np.sum(_z * _ma[_valid, _j] * _x[_valid, _j]))
            _glogb = _b[_j] * (_sb - _nj_arr[_j] / _b[_j])
            if _calib == "affine":
                _grad_a[_j] = _sa
                _grad_logb[_j] = _glogb
            else:
                _grad_a += _sa
                _grad_logb += _glogb

        if _calib == "affine":
            _gradient = np.concatenate([_grad_a, _grad_logb])
        else:
            _gradient = np.array([_grad_a, _grad_logb], dtype=float)
        return float(_nll), _gradient

    def _vsn_lts(_x, _calib, _lts_quantile, _optimpar):
        """Robust ML fitting with R-compatible LTS row selection."""
        _nr, _nc = _x.shape
        if _calib == "affine":
            _p0 = np.zeros(2 * _nc)
            _p0[_nc:] = 1.0
            _bounds = [(None, None)] * _nc + [(-100.0, 100.0)] * _nc
        else:
            _p0 = np.array([0.0, 1.0])
            _bounds = [(None, None), (-100.0, 100.0)]

        _opt_kw = {
            "maxiter": int(_optimpar["maxit"]),
            "maxcor": 5,
            "ftol": float(_optimpar["factr"]) * np.finfo(float).eps,
            "gtol": float(_optimpar["pgtol"]),
        }
        _selected = None
        _old_hy = None
        _cvg_count = 0

        for _iteration in range(1, int(_optimpar["cvg_niter"]) + 1):
            _x_fit = _x if _selected is None else _x[_selected, :]
            if _x_fit.shape[0] == 0:
                raise RuntimeError("LTS selection removed every fitting row.")
            _result = minimize(
                _negloglik_and_grad,
                _p0,
                args=(_x_fit, _calib),
                jac=True,
                method="L-BFGS-B",
                bounds=_bounds,
                options=_opt_kw,
            )
            if not _result.success:
                raise RuntimeError(
                    "L-BFGS-B failed: "
                    f"status={_result.status}, message={_result.message}"
                )
            _p0 = np.asarray(_result.x, dtype=np.float64)

            if int(_optimpar["trace"]) > 0:
                print(
                    f"LTS iteration {_iteration}: rows={_x_fit.shape[0]}, "
                    f"objective={_result.fun:.12g}, status={_result.status}, "
                    f"optimizer_iterations={_result.nit}"
                )
            if _lts_quantile >= 1.0:
                break

            _a_fit, _logb_fit = _unpack_parameters(_p0, _nc, _calib)
            _b_fit = np.exp(_logb_fit)
            _hy = np.full((_nr, _nc), np.nan)
            for _j in range(_nc):
                _valid = np.isfinite(_x[:, _j])
                if _valid.any():
                    _hy[_valid, _j] = np.arcsinh(
                        _b_fit[_j] * _x[_valid, _j] + _a_fit[_j]
                    )

            _hmean = np.nanmean(_hy, axis=1)
            _rvar = np.sum((_hy - _hmean[:, np.newaxis]) ** 2, axis=1)
            _rank_hmean = rankdata(_hmean, method="average", nan_policy="omit")
            _rank_min = float(np.nanmin(_rank_hmean))
            _rank_max = float(np.nanmax(_rank_hmean))
            _rank_span = _rank_max - _rank_min
            _cuts = np.linspace(_rank_min, _rank_max, 6)
            _cuts[0] -= _rank_span * 0.001
            _cuts[-1] += _rank_span * 0.001
            _slabs = np.searchsorted(_cuts, _rank_hmean, side="left") - 1
            _slabs = np.clip(_slabs, 0, 4)

            _selected_mask = np.zeros(_nr, dtype=bool)
            for _slice in range(5):
                _mask = _slabs == _slice
                if not _mask.any():
                    continue
                _finite_rvar = np.isfinite(_rvar[_mask])
                if not _finite_rvar.any() and _slice != 0:
                    continue
                _quantile = float(np.nanquantile(_rvar[_mask], _lts_quantile))
                _selected_mask |= _mask & (
                    (_rvar <= _quantile) | (_slabs == 0)
                )
            _selected = np.flatnonzero(_selected_mask)

            if float(_optimpar["cvg_eps"]) > 0 and _old_hy is not None:
                _change = float(np.nanmax(np.abs(_hy - _old_hy)))
                _cvg_count = _cvg_count + 1 if _change < _optimpar["cvg_eps"] else 0
                if _cvg_count >= 3:
                    break
            _old_hy = _hy

        return _p0

    def vsn2(
        x,
        calib="affine",
        lts_quantile=0.9,
        verbose=False,
        optimpar=None,
        min_data_points_per_stratum=42,
    ):
        """Standalone equivalent of ``vsn.justvsn`` for one stratum."""
        _x = np.asarray(x, dtype=np.float64)
        if _x.ndim != 2:
            raise ValueError("x must be a two-dimensional numeric matrix.")
        _nr, _nc = _x.shape
        if _nc < 2:
            raise ValueError("x needs two or more columns.")
        if calib not in {"affine", "none"}:
            raise ValueError("calib must be 'affine' or 'none'.")
        if not 0.5 <= float(lts_quantile) <= 1.0:
            raise ValueError("lts_quantile must be between 0.5 and 1.0.")
        if _nr < int(min_data_points_per_stratum):
            raise ValueError(
                f"The matrix has {_nr} rows, below the required "
                f"minimum of {int(min_data_points_per_stratum)}."
            )

        _opar = DEFAULT_OPTIMPAR.copy()
        if optimpar:
            _unknown = set(optimpar) - OPTIMPAR_NAMES
            if _unknown:
                raise ValueError(f"Unknown optimizer parameters: {sorted(_unknown)}")
            _opar.update(optimpar)

        _x[~np.isfinite(_x)] = np.nan
        _all_na = np.all(np.isnan(_x), axis=1)
        _x_fit = _x[~_all_na, :]
        if _x_fit.shape[0] == 0:
            raise ValueError("x contains no row with a finite observation.")
        if verbose:
            print(f"vsn2: {_nr} x {_nc} matrix (1 stratum).")
            if _all_na.any():
                warnings.warn(
                    f"{int(_all_na.sum())} rows were removed since they "
                    "contained only NA elements."
                )

        _p_fit = _vsn_lts(_x_fit, calib, float(lts_quantile), _opar)
        _a_fit, _logb_fit = _unpack_parameters(_p_fit, _nc, calib)
        _b_fit = np.exp(_logb_fit)
        _hoffset = np.log2(2.0 * np.exp(float(np.mean(_logb_fit))))
        _hx = np.full_like(_x, np.nan)
        for _j in range(_nc):
            _valid = np.isfinite(_x[:, _j])
            if _valid.any():
                _hx[_valid, _j] = (
                    np.arcsinh(_b_fit[_j] * _x[_valid, _j] + _a_fit[_j])
                    / np.log(2.0)
                    - _hoffset
                )
        if verbose:
            print("Please use a mean-SD plot to verify the fit.")
        return _hx

    return DEFAULT_OPTIMPAR, _negloglik_and_grad, _vsn_lts, vsn2


@app.cell(hide_code=True)
def _(mo):
    file_info = mo.md(
        "Upload proteinGroups.txt / report.tsv / csv-file<br>"
        "(e.g., <a href='https://zenodo.org/records/14557756/files/proteinGroups.txt?download=1' target='_blank'>file</a>, "
        "details in <a href='https://fuzzylife.substack.com/p/proteomics-data-processing-with-maxquant' target='_blank'>Proteomics data processing with MaxQuant</a>)"
    )
    file_input = mo.ui.file(filetypes=[".txt", ".tsv", ".csv"], label="Upload proteinGroups.txt / .tsv")
    intensity_prefix = mo.ui.text(value="LFQ", label="Intensity column prefix")
    calib_type = mo.ui.dropdown(
        options=["affine", "none"],
        value="affine",
        label="Calibration mode",
    )
    lts_quantile_input = mo.ui.number(
        start=0.5,
        stop=1.0,
        step=0.01,
        value=0.9,
        label="LTS quantile",
    )
    factr_input = mo.ui.number(
        start=1.0,
        stop=1e12,
        step=1e6,
        value=5e7,
        label="L-BFGS-B factr",
    )
    pgtol_input = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=1e-5,
        value=2e-4,
        label="Projected-gradient tolerance",
    )
    maxit_input = mo.ui.number(
        start=1,
        stop=1000000,
        step=1000,
        value=60000,
        label="Maximum optimizer iterations",
    )
    trace_input = mo.ui.number(
        start=0,
        stop=100,
        step=1,
        value=0,
        label="Optimizer trace level",
    )
    cvg_niter_input = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=7,
        label="Maximum LTS iterations",
    )
    cvg_eps_input = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=1e-4,
        value=0.0,
        label="LTS convergence epsilon",
    )
    min_rows_input = mo.ui.number(
        start=0,
        stop=1000000,
        step=1,
        value=42,
        label="Minimum rows per stratum",
    )
    verbose_input = mo.ui.checkbox(
        value=True,
        label="Print fitting progress",
    )
    run_btn = mo.ui.run_button(
        label="Run VSN Normalization",
        kind="success",
        tooltip="Execute VSN on matched columns",
    )
    return (
        calib_type,
        cvg_eps_input,
        cvg_niter_input,
        factr_input,
        file_input,
        intensity_prefix,
        lts_quantile_input,
        maxit_input,
        min_rows_input,
        pgtol_input,
        run_btn,
        trace_input,
        verbose_input,
    )


@app.cell(hide_code=True)
def _(file_input, io, pd):
    df_raw = None
    if file_input.value:
        _file = file_input.value[0]
        try:
            _sep = "\t" if _file.name.endswith((".txt", ".tsv")) else ","
            df_raw = pd.read_csv(io.BytesIO(_file.contents), sep=_sep)
        except Exception:
            pass
    return df_raw,


@app.cell(hide_code=True)
def _(df_raw, intensity_prefix, mo):
    cols_detected = []
    sample_select = None

    if df_raw is not None and intensity_prefix.value:
        cols_detected = [c for c in df_raw.columns if c.startswith(intensity_prefix.value)]
        if cols_detected:
            sample_select = mo.ui.dropdown(
                options=cols_detected,
                value=cols_detected[0],
                label="Sample for log2 regression:"
            )

    return cols_detected, sample_select


@app.cell(hide_code=True)
def _(
    calib_type,
    cols_detected,
    cvg_eps_input,
    cvg_niter_input,
    df_raw,
    datetime,
    factr_input,
    lts_quantile_input,
    maxit_input,
    min_rows_input,
    mo,
    np,
    pd,
    pgtol_input,
    run_btn,
    trace_input,
    verbose_input,
    vsn2,
):
    df_out = None
    normalized_mat = None
    generated_vsn_columns = []
    generated_pow2_columns = []
    run_timestamp = None

    if df_raw is not None and cols_detected and run_btn.value:
        with mo.status.spinner(subtitle="Running VSN Normalization…"):
            _mat = (
                df_raw[cols_detected]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64, copy=True)
            )
            _mat[~np.isfinite(_mat)] = np.nan
            _mat[_mat <= 0] = np.nan

            _ui_optimpar = {
                "factr": float(factr_input.value),
                "pgtol": float(pgtol_input.value),
                "maxit": int(maxit_input.value),
                "trace": int(trace_input.value),
                "cvg_niter": int(cvg_niter_input.value),
                "cvg_eps": float(cvg_eps_input.value),
            }

            # vsn2() returns the correctly scaled log2 matrix directly.
            normalized_mat = vsn2(
                _mat,
                calib=calib_type.value,
                lts_quantile=float(lts_quantile_input.value),
                verbose=bool(verbose_input.value),
                optimpar=_ui_optimpar,
                min_data_points_per_stratum=int(min_rows_input.value),
            )

            _base_vsn = [f"vsn_{c}" for c in cols_detected]
            _base_pow2 = [f"pow2_{c}" for c in cols_detected]
            if any(c in df_raw.columns for c in _base_vsn + _base_pow2):
                run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                generated_vsn_columns = [f"{c}_{run_timestamp}" for c in _base_vsn]
                generated_pow2_columns = [f"{c}_{run_timestamp}" for c in _base_pow2]
            else:
                generated_vsn_columns = _base_vsn
                generated_pow2_columns = _base_pow2
            df_norm = pd.DataFrame(normalized_mat, columns=generated_vsn_columns, index=df_raw.index)
            df_pow2 = pd.DataFrame(2.0 ** normalized_mat, columns=generated_pow2_columns, index=df_raw.index)
            df_out = pd.concat([df_raw.copy(), df_norm, df_pow2], axis=1)

    return df_out, generated_pow2_columns, generated_vsn_columns, normalized_mat, run_timestamp


@app.cell(hide_code=True)
def _(
    calib_type,
    cols_detected,
    cvg_eps_input,
    cvg_niter_input,
    df_out,
    df_raw,
    factr_input,
    generated_pow2_columns,
    generated_vsn_columns,
    file_input,
    intensity_prefix,
    io,
    linregress,
    lts_quantile_input,
    maxit_input,
    min_rows_input,
    mo,
    normalized_mat,
    np,
    Path,
    pgtol_input,
    plt,
    run_btn,
    run_timestamp,
    sample_select,
    trace_input,
    verbose_input,
):

    # Keep run and download actions together in the sidebar.
    if df_out is not None:
        _tsv_buf = io.BytesIO()
        df_out.to_csv(_tsv_buf, index=False, sep="\t")
        _stem = Path(file_input.value[0].name).stem
        _suffix = (
            f"vsn_calib-{calib_type.value}_lts-{float(lts_quantile_input.value):g}"
            f"_factr-{float(factr_input.value):g}_pgtol-{float(pgtol_input.value):g}"
            f"_maxit-{int(maxit_input.value)}_trace-{int(trace_input.value)}"
            f"_ltsiter-{int(cvg_niter_input.value)}_cvgeps-{float(cvg_eps_input.value):g}"
            f"_minrows-{int(min_rows_input.value)}"
        ).replace("+", "").replace(".", "p")
        _safe_prefix = "".join(
            _character if _character.isalnum() or _character in "-_" else "-"
            for _character in intensity_prefix.value.strip()
        ).strip("-_") or "intensity"
        _download_filename = f"{_stem}_{_safe_prefix}_{_suffix}.txt"
        _download_button = mo.download(
            data=_tsv_buf.getvalue(),
            filename=_download_filename,
            label="Download VSN Normalized Data",
            mimetype="text/tab-separated-values",
        )
        _run_actions = mo.vstack([run_btn, _download_button], gap=0.5)
        mo.status.toast(
            title="Normalization complete",
            description=f"Successfully normalized {len(cols_detected)} columns.",
        )
    else:
        _run_actions = run_btn

    # --- Sidebar ---
    _sidebar = mo.vstack([
        mo.md("### Options"),
        file_info,      # This references the new mo.md block containing the links
        file_input,
        intensity_prefix,
        _run_actions,
        mo.md("---"),
        mo.md("#### VSN parameters"),
        calib_type,
        lts_quantile_input,
        mo.md("#### Optimizer parameters"),
        factr_input,
        pgtol_input,
        maxit_input,
        trace_input,
        mo.md("#### LTS convergence"),
        cvg_niter_input,
        cvg_eps_input,
        min_rows_input,
        verbose_input,
    ], gap=1).style({"padding": "1rem", "background": "#f8f9fa", "border-radius": "8px"})

    # --- Main panel routing ---
    if df_raw is None:
        _main = mo.md(
            "### Welcome to the VSN Dashboard\n\n"
            "Upload a `proteinGroups.txt` or similar file to begin."
        )

    elif not cols_detected:
        _main = mo.md(
            f"⚠️ No columns matched prefix `{intensity_prefix.value}`. "
            "Check headers or adjust the prefix."
        )

    elif df_out is None:
        _main = mo.vstack([
            mo.md(f"✅ **File loaded:** `{file_input.value[0].name}`"),
            mo.md(f"✅ **Matched {len(cols_detected)} intensity columns.**"),
            mo.md("👉 Click **Run VSN Normalization** to process."),
        ])

    else:
        # ── Preview tab ──────────────────────────────────────────────────────
        _preview_ui = mo.vstack([
            mo.md(f"#### Normalized Dataset Preview ({df_out.shape[0]} rows × {df_out.shape[1]} cols)"),
            mo.ui.table(df_out.head(15), selection=None),
        ])

        # ── Diagnostics tab ──────────────────────────────────────────────────
        # Mean-SD, VSN distribution, and before/after sample boxplots.
        _fig, _axes = plt.subplots(1, 4, figsize=(20, 4.5))

        _finite_counts = np.sum(np.isfinite(normalized_mat), axis=1)
        _row_means = np.full(normalized_mat.shape[0], np.nan)
        _row_sds = np.full(normalized_mat.shape[0], np.nan)

        _mean_rows = _finite_counts >= 1
        _sd_rows = _finite_counts >= 2
        if np.any(_mean_rows):
            _row_means[_mean_rows] = np.nanmean(
                normalized_mat[_mean_rows], axis=1
            )
        if np.any(_sd_rows):
            _row_sds[_sd_rows] = np.nanstd(
                normalized_mat[_sd_rows], axis=1, ddof=1
            )

        _valid = np.isfinite(_row_means) & np.isfinite(_row_sds)

        if np.any(_valid):
            _xm  = _row_means[_valid]
            _ysd = _row_sds[_valid]

            # ── Left plot: mean-SD with RANKS on x-axis (matches R ranks=TRUE) ──
            from scipy.stats import rankdata as _rd
            _ranks = _rd(_xm, method="average")

            _axes[0].scatter(_ranks, _ysd, s=1, alpha=0.15, color="grey")

            # Running median (50 bins of equal rank width)
            _n_bins     = min(50, max(3, len(_ranks) // 20))
            _bin_edges  = np.linspace(_ranks.min(), _ranks.max(), _n_bins + 1)
            _bin_ctrs   = 0.5 * (_bin_edges[:-1] + _bin_edges[1:])
            _bin_meds   = []
            for _i in range(_n_bins):
                _bmask = (_ranks >= _bin_edges[_i]) & (_ranks < _bin_edges[_i + 1])
                _bin_meds.append(np.nanmedian(_ysd[_bmask]) if _bmask.any() else np.nan)
            _axes[0].plot(_bin_ctrs, _bin_meds, color="red", linewidth=2, label="running median")
            _axes[0].legend(fontsize=8)

        _axes[0].set_title("Mean-SD Plot  (ranks=TRUE)")
        _axes[0].set_xlabel("Rank(Row Mean)")
        _axes[0].set_ylabel("Row Standard Deviation")

        # ── Middle plot: histogram of all values ──────────────────────────────
        _flat = normalized_mat[np.isfinite(normalized_mat)].flatten()
        _axes[1].hist(_flat, bins=60, color="darkgray", edgecolor="white")
        _axes[1].set_title("Distribution of VSN Values")
        _axes[1].set_xlabel("VSN (log2 scale)")
        _axes[1].set_ylabel("Frequency")

        _short_names = [
            c.replace(intensity_prefix.value, "").lstrip("_. ") or f"S{i+1}"
            for i, c in enumerate(cols_detected)
        ]

        # Raw LFQ boxplot before VSN, on log2 scale for comparability.
        _raw_matrix = df_raw[cols_detected].to_numpy(dtype=float, copy=True)
        _raw_matrix[_raw_matrix <= 0] = np.nan
        _raw_log2 = np.log2(_raw_matrix)
        _raw_bp_data = [
            _raw_log2[:, j][np.isfinite(_raw_log2[:, j])]
            for j in range(_raw_log2.shape[1])
        ]
        _axes[2].boxplot(
            _raw_bp_data,
            patch_artist=True,
            boxprops=dict(facecolor="wheat"),
            flierprops=dict(marker=".", markersize=1, alpha=0.3),
            showfliers=False,
        )
        _axes[2].set_title("Raw LFQ before VSN")
        _axes[2].set_ylabel("log2(LFQ intensity)")
        _axes[2].set_xticks(range(1, len(_short_names) + 1))
        _axes[2].set_xticklabels(_short_names, rotation=45, ha="right", fontsize=7)

        # VSN-normalized boxplot per sample.
        _bp_data = [
            normalized_mat[:, j][np.isfinite(normalized_mat[:, j])]
            for j in range(normalized_mat.shape[1])
        ]
        _axes[3].boxplot(
            _bp_data,
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue"),
            flierprops=dict(marker=".", markersize=1, alpha=0.3),
            showfliers=False,
        )
        _axes[3].set_title("VSN normalized per sample")
        _axes[3].set_ylabel("VSN (log2 scale)")
        _axes[3].set_xticks(range(1, len(_short_names) + 1))
        _axes[3].set_xticklabels(_short_names, rotation=45, ha="right", fontsize=7)

        plt.tight_layout()
        _diagnostics_ui = mo.as_html(_fig)
        plt.close(_fig)

        # ── Regression tab ────────────────────────────────────────────────────
        if sample_select is not None and sample_select.value:
            _sel = sample_select.value
            _raw_v = df_raw[_sel].values.copy().astype(float)
            _raw_v[_raw_v <= 0] = np.nan
            _selected_vsn_column = generated_vsn_columns[cols_detected.index(_sel)]
            _vsn_v = df_out[_selected_vsn_column].values
            _log2_v = np.log2(_raw_v)
            _mval = np.isfinite(_log2_v) & np.isfinite(_vsn_v)

            if _mval.sum() > 5:
                _sl, _ic, _rv, _pv, _se = linregress(_log2_v[_mval], _vsn_v[_mval])
                _fig_r, _ax_r = plt.subplots(figsize=(6, 4))
                _ax_r.scatter(_log2_v[_mval], _vsn_v[_mval], s=2, alpha=0.3, color="steelblue")
                _lx = np.linspace(_log2_v[_mval].min(), _log2_v[_mval].max(), 200)
                _ax_r.plot(_lx, _sl * _lx + _ic, color="red", linewidth=1.5,
                           label=f"slope={_sl:.4f}  R²={_rv**2:.4f}\nintercept={_ic:.4f}")
                _ax_r.set_title(f"VSN vs log2: {_sel}")
                _ax_r.set_xlabel(f"log2({_sel})")
                _ax_r.set_ylabel(_selected_vsn_column)
                _ax_r.legend(loc="upper left", fontsize=8)
                plt.tight_layout()
                _regression_plot_ui = mo.as_html(_fig_r)
                plt.close(_fig_r)
            else:
                _regression_plot_ui = mo.md("Insufficient data points for regression.")

            _regression_ui = mo.vstack([
                mo.hstack([sample_select], justify="start"),
                _regression_plot_ui,
            ])
        else:
            _regression_ui = mo.md("Select a sample to view regression.")

        _tabs = mo.ui.tabs({
            "Preview":     _preview_ui,
            "Diagnostics": _diagnostics_ui,
            "Regression":  _regression_ui,
        })

        _main = _tabs

    dashboard = mo.vstack([
        mo.md("# Variance Stabilizing Normalization (VSN) Dashboard"),
        mo.hstack([_sidebar, _main], widths=[1, 3], gap=2),
    ])
    dashboard
    return dashboard,


if __name__ == "__main__":
    app.run()