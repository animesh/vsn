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
    from dataclasses import dataclass, field
    from typing import Optional

    return Optional, dataclass, field, io, linregress, minimize, mo, np, pd, plt, rankdata, warnings


@app.cell(hide_code=True)
def _(dataclass, field, Optional, np, minimize, rankdata, warnings):
    # ---------------------------------------------------------------------------
    # Embedded vsn2.py 
    # ---------------------------------------------------------------------------

    OPTIMPAR_NAMES = {"factr", "pgtol", "maxit", "trace", "cvg_niter", "cvg_eps"}

    DEFAULT_OPTIMPAR = {
        "factr": 5e7,
        "pgtol": 2e-4,
        "maxit": 60000,
        "trace": 0,
        "cvg_niter": 7,
        "cvg_eps": 0.0,
    }

    @dataclass
    class VsnResult:
        coefficients: np.ndarray 
        strata: np.ndarray       
        calib: str              
        hoffset: float = 0.0      

        def predict(self, x: np.ndarray) -> np.ndarray:
            nr, nc = x.shape
            res = np.empty_like(x, dtype=float)
            unique_strata = np.unique(self.strata)
            for s_idx, s in enumerate(unique_strata):
                mask = (self.strata == s)
                if not np.any(mask):
                    continue
                xs = x[mask, :]
                ys = np.empty_like(xs, dtype=float)
                for j in range(nc):
                    p = self.coefficients[s_idx, j, :]
                    if self.calib == "affine":
                        a, b = p[0], p[1]
                        ys[:, j] = np.arcsinh(a + b * xs[:, j])
                    elif self.calib == "none":
                        b = p[0]
                        ys[:, j] = np.arcsinh(b * xs[:, j])
                res[mask, :] = ys
            return res - self.hoffset

    @dataclass
    class VsnInput:
        x: np.ndarray
        reference: Optional[VsnResult] = None
        strata: np.ndarray = field(default_factory=lambda: np.array([0]))
        ordered: bool = False
        lts_quantile: float = 0.9
        subsample: Optional[int] = None
        verbose: bool = False
        calib: str = "affine"
        pstart: Optional[np.ndarray] = None
        optimpar: dict = field(default_factory=lambda: DEFAULT_OPTIMPAR.copy())

    def scaling_factor_transformation(b: np.ndarray) -> float:
        return float(np.nanmean(np.log2(b)))

    def pstart_heuristic(x: np.ndarray, strata_split: list[np.ndarray], calib: str) -> np.ndarray:
        nr, nc = x.shape
        n_strata = len(strata_split)
        n_params = 2 if calib == "affine" else 1
        pstart = np.zeros((n_strata, nc, n_params))
        for s_idx, mask in enumerate(strata_split):
            xs = x[mask, :]
            for j in range(nc):
                col = xs[:, j]
                col_valid = col[col > 0]
                if len(col_valid) == 0:
                    b_init = 1.0
                else:
                    b_init = 1.0 / (np.nanmedian(col_valid) + 1e-8)
                if calib == "affine":
                    pstart[s_idx, j, 0] = 0.0      
                    pstart[s_idx, j, 1] = b_init   
                else:
                    pstart[s_idx, j, 0] = b_init   
        return pstart

    def vsn_obj_func(p: np.ndarray, x: np.ndarray, calib: str, ref: Optional[VsnResult] = None) -> tuple[float, np.ndarray]:
        nr, nc = x.shape
        n_params = 2 if calib == "affine" else 1
        p_mat = p.reshape((nc, n_params))
        
        y = np.empty_like(x, dtype=float)
        for j in range(nc):
            pj = p_mat[j, :]
            if calib == "affine":
                a, b = pj[0], pj[1]
                y[:, j] = np.arcsinh(a + b * x[:, j])
            elif calib == "none":
                b = pj[0]
                y[:, j] = np.arcsinh(b * x[:, j])

        row_means = np.nanmean(y, axis=1, keepdims=True)
        res = y - row_means
        ss = np.nansum(res**2, axis=1)
        ss = np.maximum(ss, 1e-12)
        
        log_jac = np.zeros_like(x, dtype=float)
        for j in range(nc):
            pj = p_mat[j, :]
            if calib == "affine":
                a, b = pj[0], pj[1]
                val = a + b * x[:, j]
                log_jac[:, j] = np.log(np.abs(b) + 1e-12) - 0.5 * np.log(1.0 + val**2)
            elif calib == "none":
                b = pj[0]
                val = b * x[:, j]
                log_jac[:, j] = np.log(np.abs(b) + 1e-12) - 0.5 * np.log(1.0 + val**2)
                
        nll = 0.5 * nc * np.nansum(np.log(ss)) - np.nansum(log_jac)
        
        grad = np.zeros_like(p_mat)
        for j in range(nc):
            pj = p_mat[j, :]
            if calib == "affine":
                a, b = pj[0], pj[1]
                val = a + b * x[:, j]
                sqrt_val = np.sqrt(1.0 + val**2)
                
                dy_da = 1.0 / sqrt_val
                dy_db = x[:, j] / sqrt_val
                
                dlog_da = - val / (1.0 + val**2)
                dlog_db = 1.0 / (b + 1e-12) - x[:, j] * val / (1.0 + val**2)
                
                dnll_dy = nc * res[:, j] / ss
                grad[j, 0] = np.nansum(dnll_dy * dy_da) - np.nansum(dlog_da)
                grad[j, 1] = np.nansum(dnll_dy * dy_db) - np.nansum(dlog_db)
            elif calib == "none":
                b = pj[0]
                val = b * x[:, j]
                sqrt_val = np.sqrt(1.0 + val**2)
                
                dy_db = x[:, j] / sqrt_val
                dlog_db = 1.0 / (b + 1e-12) - x[:, j] * val / (1.0 + val**2)
                
                dnll_dy = nc * res[:, j] / ss
                grad[j, 0] = np.nansum(dnll_dy * dy_db) - np.nansum(dlog_db)
                
        return nll, grad.flatten()

    def vsn_sample(v: VsnInput) -> VsnResult:
        x = v.x
        nr, nc = x.shape
        
        if len(v.strata) == 1 and v.strata[0] == 0:
            strata_arr = np.zeros(nr, dtype=int)
        else:
            strata_arr = v.strata
            
        unique_strata = np.unique(strata_arr)
        n_strata = len(unique_strata)
        strata_split = [np.where(strata_arr == s)[0] for s in unique_strata]
        
        pstart = v.pstart
        if pstart is None:
            pstart = pstart_heuristic(x, strata_split, v.calib)
            
        n_params = 2 if v.calib == "affine" else 1
        coefficients = np.zeros((n_strata, nc, 2))
        
        for s_idx, mask in enumerate(strata_split):
            xs = x[mask, :]
            if v.subsample is not None and v.subsample < len(mask):
                idx = np.random.choice(len(mask), v.subsample, replace=False)
                xs_sub = xs[idx, :]
            else:
                xs_sub = xs
                
            p0 = pstart[s_idx].flatten()
            bounds = []
            for j in range(nc):
                if v.calib == "affine":
                    bounds.append((None, None))
                    bounds.append((1e-10, None))
                else:
                    bounds.append((1e-10, None))
                    
            res_opt = minimize(
                fun=vsn_obj_func,
                x0=p0,
                args=(xs_sub, v.calib, v.reference),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": v.optimpar.get("maxit", 60000),
                    "ftol": v.optimpar.get("factr", 5e7) * 2.220446049250313e-16,
                    "gtol": v.optimpar.get("pgtol", 2e-4),
                    "disp": v.verbose
                }
            )
            
            p_fit = res_opt.x.reshape((nc, n_params))
            for j in range(nc):
                if v.calib == "affine":
                    coefficients[s_idx, j, 0] = p_fit[j, 0]
                    coefficients[s_idx, j, 1] = p_fit[j, 1]
                else:
                    coefficients[s_idx, j, 0] = 0.0
                    coefficients[s_idx, j, 1] = p_fit[j, 0]
                    
        return VsnResult(coefficients=coefficients, strata=strata_arr, calib=v.calib)

    def vsn2(
        x: np.ndarray,
        reference: Optional[VsnResult] = None,
        strata: Optional[np.ndarray] = None,
        calib: str = "affine",
        lts_quantile: float = 0.9,
        subsample: Optional[int] = None,
        verbose: bool = False,
        pstart: Optional[np.ndarray] = None,
        optimpar: Optional[dict] = None,
    ) -> VsnResult:
        nr, nc = x.shape
        if strata is None:
            strata_arr = np.zeros(nr, dtype=int)
        else:
            strata_arr = strata
            
        opar = DEFAULT_OPTIMPAR.copy()
        if optimpar is not None:
            opar.update(optimpar)
            
        v_in = VsnInput(
            x=x,
            reference=reference,
            strata=strata_arr,
            ordered=False,
            lts_quantile=lts_quantile,
            subsample=subsample,
            verbose=verbose,
            calib=calib,
            pstart=pstart,
            optimpar=opar,
        )
        
        res = vsn_sample(v_in)
        b_vals = res.coefficients[:, :, 1]
        b_mean = np.nanmean(b_vals)
        res.hoffset = np.log2(2.0 * b_mean) if b_mean > 0 else 0.0
        return res

    return DEFAULT_OPTIMPAR, VsnInput, VsnResult, pstart_heuristic, scaling_factor_transformation, vsn2, vsn_obj_func, vsn_sample


@app.cell(hide_code=True)
def _(mo):
    file_input = mo.ui.file(filetypes=[".txt", ".tsv", ".csv"], label="Upload proteinGroups.txt / .tsv")
    intensity_prefix = mo.ui.text(value="LFQ.intensity", label="Intensity column prefix")
    calib_type = mo.ui.dropdown(options=["affine", "none"], value="affine", label="Calibration Mode")
    run_btn = mo.ui.run_button(label="Run VSN Normalization", kind="success", tooltip="Execute VSN on matched columns")
    return calib_type, file_input, intensity_prefix, run_btn


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
def _(calib_type, cols_detected, df_raw, mo, np, pd, run_btn, vsn2):
    df_out = None
    normalized_mat = None

    if df_raw is not None and cols_detected and run_btn.value:
        with mo.status.spinner(subtitle="Running VSN Normalization (this may take a few seconds)..."):
            _mat = df_raw[cols_detected].values.copy().astype(float)
            _mat[_mat <= 0] = np.nan

            _vsn_fit = vsn2(_mat, calib=calib_type.value, verbose=False)
            normalized_mat = _vsn_fit.predict(_mat)

            _meta_cols = [c for c in df_raw.columns if c not in cols_detected]
            df_meta = df_raw[_meta_cols].copy()
            df_norm = pd.DataFrame(normalized_mat, columns=[f"vsn_{c}" for c in cols_detected], index=df_raw.index)
            df_pow2 = pd.DataFrame(2.0**normalized_mat, columns=[f"pow2_{c}" for c in cols_detected], index=df_raw.index)

            df_out = pd.concat([df_meta, df_raw[cols_detected], df_norm, df_pow2], axis=1)

    return df_out, normalized_mat


@app.cell(hide_code=True)
def _(calib_type, cols_detected, df_out, df_raw, file_input, intensity_prefix, io, linregress, mo, normalized_mat, np, plt, run_btn, sample_select):
    # --- Sidebar Layout Construction ---
    _sidebar = mo.vstack([
        mo.md("### Options"),
        file_input,
        intensity_prefix,
        calib_type,
        mo.md("---"),
        run_btn
    ], gap=1).style({"padding": "1rem", "background": "#f8f9fa", "border-radius": "8px"})

    # --- Main Panel Routing ---
    if df_raw is None:
        _main = mo.md("### Welcome to the VSN Dashboard\n\nPlease upload a `proteinGroups.txt` or similar dataset using the sidebar to begin.")
        
    elif not cols_detected:
        _main = mo.md(f"⚠️ **Dataset loaded, but no columns matched the prefix `{intensity_prefix.value}`.**\n\nPlease check your file headers or adjust the prefix input.")
        
    elif df_out is None:
        _main = mo.vstack([
            mo.md(f"✅ **File Loaded:** `{file_input.value[0].name}`"),
            mo.md(f"✅ **Matched {len(cols_detected)} intensity columns.**"),
            mo.md("👉 **Click the 'Run VSN Normalization' button in the sidebar to process the data.**")
        ])
        
    else:
        # --- Results Layout Construction ---
        _preview_ui = mo.vstack([
            mo.md(f"#### Normalized Dataset Preview ({df_out.shape[0]} rows x {df_out.shape[1]} cols)"),
            mo.ui.table(df_out.head(15), selection=None)
        ])

        _fig, _ax = plt.subplots(1, 2, figsize=(11, 4.5))
        _row_means = np.nanmean(normalized_mat, axis=1)
        _row_sds = np.nanstd(normalized_mat, axis=1)
        _valid = np.isfinite(_row_means) & np.isfinite(_row_sds) & (_row_sds > 0)
        
        if np.any(_valid):
            _xm = _row_means[_valid]
            _ysd = _row_sds[_valid]
            _ax[0].scatter(_xm, _ysd, s=1, alpha=0.15, color="black")
            _n_bins = min(50, len(_xm) // 10)
            if _n_bins > 2:
                _bin_edges = np.linspace(_xm.min(), _xm.max(), _n_bins + 1)
                _bin_centers = 0.5 * (_bin_edges[:-1] + _bin_edges[1:])
                _bin_medians = []
                for _i in range(_n_bins):
                    _b_mask = (_xm >= _bin_edges[_i]) & (_xm < _bin_edges[_i+1])
                    _bin_medians.append(np.nanmedian(_ysd[_b_mask]) if np.any(_b_mask) else np.nan)
                _ax[0].plot(_bin_centers, _bin_medians, color="red", linewidth=2, label="running median")
                _ax[0].legend()
                
        _ax[0].set_title("Mean-SD Diagnostics Plot")
        _ax[0].set_xlabel("Stabilized Mean")
        _ax[0].set_ylabel("Row Standard Deviation")

        _ax[1].hist(normalized_mat[np.isfinite(normalized_mat)].flatten(), bins=50, color="darkgray", edgecolor="white")
        _ax[1].set_title("Histogram of Normalized Values")
        _ax[1].set_xlabel("Stabilized value scale (vsn)")
        _ax[1].set_ylabel("Frequency")
        plt.tight_layout()
        
        _diagnostics_ui = mo.as_html(_fig)
        plt.close(_fig)

        # Embedded Dropdown for Regression
        if sample_select is not None and sample_select.value:
            _selected_col = sample_select.value
            _raw_v = df_raw[_selected_col].values.copy().astype(float)
            _vsn_v = df_out[f"vsn_{_selected_col}"].values
            _log2_v = np.log2(_raw_v)
            _m_val = np.isfinite(_log2_v) & np.isfinite(_vsn_v)

            if np.sum(_m_val) > 5:
                _slope, _intercept, _r_val, _p_val, _se = linregress(_log2_v[_m_val], _vsn_v[_m_val])
                _fig_reg, _ax_reg = plt.subplots(figsize=(6, 4))
                _ax_reg.scatter(_log2_v[_m_val], _vsn_v[_m_val], s=2, alpha=0.3, color="blue")
                _lx = np.linspace(_log2_v[_m_val].min(), _log2_v[_m_val].max(), 100)
                _ax_reg.plot(_lx, _slope * _lx + _intercept, color="red", linewidth=1.5, label=f"R² = {_r_val**2:.4f}")
                _ax_reg.set_title(f"VSN vs log2: {_selected_col}")
                _ax_reg.set_xlabel(f"log2({_selected_col})")
                _ax_reg.set_ylabel(f"vsn_{_selected_col}")
                _ax_reg.legend(loc="upper left")
                plt.tight_layout()
                _regression_plot_ui = mo.as_html(_fig_reg)
                plt.close(_fig_reg)
            else:
                _regression_plot_ui = mo.md("Insufficient overlapping data points to run linear fit.")

            _regression_ui = mo.vstack([
                mo.hstack([sample_select], justify="start"),
                _regression_plot_ui
            ])
        else:
            _regression_ui = mo.md("Select a sample to view regression.")

        _tsv_buf = io.BytesIO()
        df_out.to_csv(_tsv_buf, index=False, sep="\t")
        _download_btn = mo.download(
            data=_tsv_buf.getvalue(),
            filename="normalized_vsn_data.txt",
            label="📥 Download VSN Normalized Data",
            mimetype="text/tab-separated-values"
        )

        _tabs = mo.ui.tabs({
            "Preview": _preview_ui,
            "Diagnostics": _diagnostics_ui,
            "Regression": _regression_ui
        })

        _main = mo.vstack([
            mo.hstack([mo.md(f"✅ Successfully normalized **{len(cols_detected)}** columns."), _download_btn], justify="space-between", align="center"),
            mo.md("---"),
            _tabs
        ])

    dashboard = mo.vstack([
        mo.md("# Variance Stabilizing Normalization (VSN) Dashboard"),
        mo.hstack([_sidebar, _main], widths=[1, 3], gap=2)
    ])

    dashboard
    return dashboard,


if __name__ == "__main__":
    app.run()