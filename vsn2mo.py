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
#http://10.20.93.118:2718?access_token=-qW8IZTi3PwgFHbAErqa3Q


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
def _(minimize, np, rankdata, warnings):
    # ---------------------------------------------------------------------------
    # Corrected VSN implementation matching R vsn2.c exactly.
    #
    # Bugs fixed vs original marimo vsn2mo.py:
    #   Bug 1: predict() was missing / np.log(2) → output was ~32-35, not 6-22
    #   Bug 2: NLL used sum-of-logs-of-per-row-SS (wrong); now log-of-total-SS
    #   Bug 3: gradient used per-row weights; now uses scalar 1/sigsq (vsn2.c)
    #   Bug 4: no LTS robustification; now 7 iterations matching R vsnLTS
    #   Bug 5: pstart used 1/median; now log_b=1 matching R pstartHeuristic
    #   Bug 6: hoffset used arithmetic mean of b; now geometric mean (log2(2*exp(mean(log_b))))
    #   Bug 7: log_jac mixed parameterizations; now uses log_b consistently
    # ---------------------------------------------------------------------------

    DEFAULT_OPTIMPAR = {
        "factr":     5e7,
        "pgtol":     2e-4,
        "maxit":     60000,
        "trace":     0,
        "cvg_niter": 7,
        "cvg_eps":   0.0,
    }

    def _negloglik_and_grad(p, x, calib):
        """Profile NLL and gradient — exact port of vsn2.c loglik / grad_loglik.

        Parameterisation: p = [a_0..a_{nc-1}, log_b_0..log_b_{nc-1}]  (affine)
                       or p = [log_b_0..log_b_{nc-1}]                   (none)
        b_j = exp(log_b_j)  — always positive, unconstrained optimisation.
        """
        nr, nc = x.shape

        if calib == "affine":
            a    = p[:nc]
            logb = p[nc:]
        else:
            a    = np.zeros(nc)
            logb = p

        b = np.exp(logb)

        # 1st sweep: compute h_ij = arcsinh(b_j*y_ij + a_j), m_ij, Jacobian terms
        h    = np.full((nr, nc), np.nan)
        ly   = np.full((nr, nc), np.nan)   # Y_ki in vignette: b*y + a
        ma   = np.full((nr, nc), np.nan)   # A_ki: 1/sqrt(1+u^2)
        jac1 = 0.0                          # Σ log(1 + u^2)
        jac2 = 0.0                          # Σ n_j * log_b_j
        nj_arr = np.zeros(nc, dtype=int)

        for j in range(nc):
            col   = x[:, j]
            valid = np.isfinite(col)
            nj    = int(valid.sum())
            nj_arr[j] = nj
            if nj == 0:
                continue
            u           = b[j] * col[valid] + a[j]
            h[valid, j] = np.arcsinh(u)
            ly[valid, j]= u
            ma[valid, j]= 1.0 / np.sqrt(1.0 + u ** 2)
            jac1       += float(np.sum(np.log1p(u ** 2)))
            jac2       += nj * logb[j]

        jacobian = 0.5 * jac1 - jac2

        # 2nd sweep: row means → residuals → total SSQ
        mu    = np.nanmean(h, axis=1)        # profile MLE of mu_i
        resid = h - mu[:, np.newaxis]        # NaN where h is NaN

        ssq   = float(np.nansum(resid ** 2))
        nt    = int(np.sum(np.isfinite(x)))
        if nt == 0 or ssq <= 0:
            return 1e15, np.zeros_like(p)
        sigsq = ssq / nt

        # Profile NLL (vsn2.c: scale + residuals + jacobian)
        nll = (nt / 2.0) * np.log(2.0 * np.pi * sigsq) + (nt / 2.0) + jacobian

        # Gradient (vsn2.c grad_loglik)
        rfac = 1.0 / sigsq
        if calib == "affine":
            grad = np.zeros(2 * nc)
        else:
            grad = np.zeros(nc)

        for j in range(nc):
            valid = np.isfinite(x[:, j]) & np.isfinite(resid[:, j])
            if not valid.any():
                continue
            r_j  = resid[valid, j]
            m_j  = ma[valid, j]
            u_j  = ly[valid, j]
            y_j  = x[valid, j]
            nj   = nj_arr[j]

            # z = r/σ² + m*u  (vsn2.c: z = resid*rfac + ma*ly)
            z  = r_j * rfac + m_j * u_j
            sa = float(np.sum(z * m_j))
            sb = float(np.sum(z * m_j * y_j))

            if calib == "affine":
                grad[j]      = sa
                grad[nc + j] = b[j] * (sb - nj / b[j])   # chain rule for log_b
            else:
                grad[j]      = b[j] * (sb - nj / b[j])

        return nll, grad


    def _vsn_lts(x, calib, lts_quantile, optimpar):
        """Replicate R's vsnLTS: up to cvg_niter iterations of ML + LTS trimming."""
        nr, nc = x.shape

        # Starting parameters: log_b = 1 for all columns, a = 0
        # This matches R's pstartHeuristic (pstart[,,2] = 1 → log_b = 1)
        if calib == "affine":
            p0 = np.zeros(2 * nc)
            p0[nc:] = 1.0          # log_b = 1 → b = exp(1) ≈ 2.718
        else:
            p0 = np.ones(nc)       # log_b = 1

        # Bounds: a unbounded, log_b ∈ [-100, 100]  (vsn2.c)
        if calib == "affine":
            bounds = [(None, None)] * nc + [(-100.0, 100.0)] * nc
        else:
            bounds = [(-100.0, 100.0)] * nc

        cvg_niter = optimpar.get("cvg_niter", 7)
        cvg_eps   = optimpar.get("cvg_eps",   0.0)
        opt_kw    = dict(maxiter=optimpar.get("maxit", 60000),
                         ftol=optimpar.get("factr", 5e7) * np.finfo(float).eps,
                         gtol=optimpar.get("pgtol", 2e-4))

        whsel   = None     # selected row indices (None = all)
        old_hy  = None
        cvg_cnt = 0

        for iter_idx in range(1, cvg_niter + 1):
            x_fit = x if whsel is None else x[whsel, :]

            res = minimize(_negloglik_and_grad, p0,
                           args=(x_fit, calib), jac=True,
                           method="L-BFGS-B", bounds=bounds,
                           options=opt_kw)
            p0 = res.x   # warm-start next iteration

            # If LTS quantile is 1, skip row selection entirely
            if lts_quantile >= 1.0:
                break

            # Apply fitted params to ALL data to compute residuals for trimming
            if calib == "affine":
                a_fit = p0[:nc];  logb_fit = p0[nc:]
            else:
                a_fit = np.zeros(nc);  logb_fit = p0
            b_fit = np.exp(logb_fit)

            hy = np.full((nr, nc), np.nan)
            for j in range(nc):
                valid = np.isfinite(x[:, j])
                if valid.any():
                    u = b_fit[j] * x[valid, j] + a_fit[j]
                    hy[valid, j] = np.arcsinh(u)

            hmean = np.nanmean(hy, axis=1)                      # (nr,)
            rvar  = np.nansum((hy - hmean[:, np.newaxis]) ** 2, axis=1)  # (nr,)

            # LTS row selection: 5 intensity slices, keep lts_quantile of each
            # (matches R vsnLTS: cut(rank(hmean), breaks=5) + tapply quantile)
            n_slices    = 5
            valid_hm    = np.isfinite(hmean)
            rank_hmean  = np.full(nr, np.nan)
            if valid_hm.sum() > 1:
                rank_hmean[valid_hm] = rankdata(hmean[valid_hm], method="average")

            rm_min  = float(np.nanmin(rank_hmean))
            rm_max  = float(np.nanmax(rank_hmean))
            rm_span = rm_max - rm_min
            cuts    = np.linspace(rm_min - rm_span * 0.001,
                                  rm_max + rm_span * 0.001, n_slices + 1)
            slabs   = np.searchsorted(cuts, rank_hmean, side="right") - 1
            slabs   = np.clip(slabs, 0, n_slices - 1)

            whsel_mask = np.zeros(nr, dtype=bool)
            for sl in range(n_slices):
                mask = (slabs == sl)
                if not mask.any():
                    continue
                q    = float(np.nanquantile(rvar[mask], lts_quantile))
                keep = mask & ((rvar <= q) | (slabs == 0))
                whsel_mask |= keep
            whsel = np.where(whsel_mask)[0]

            # Convergence check (R's cvg.eps criterion)
            if cvg_eps > 0 and old_hy is not None:
                cvgc    = float(np.nanmax(np.abs(hy - old_hy)))
                cvg_cnt = (cvg_cnt + 1) if cvgc < cvg_eps else 0
                if cvg_cnt >= 3:
                    break
            old_hy = hy

        return p0


    def vsn2(x, calib="affine", lts_quantile=0.9,
             verbose=False, optimpar=None):
        """VSN normalization — faithful Python port of R's justvsn / vsnMatrix.

        Parameters
        ----------
        x           : (n_proteins, n_samples) float array; 0 / NaN = missing
        calib       : 'affine' (per-column a + b) or 'none' (global b)
        lts_quantile: fraction of rows used in robust LTS fit (default 0.9)
        verbose     : print progress
        optimpar    : override DEFAULT_OPTIMPAR keys

        Returns
        -------
        hx : (n_proteins, n_samples) array on log2 / glog2 scale,
             same as R justvsn output.
        """
        opar = DEFAULT_OPTIMPAR.copy()
        if optimpar:
            opar.update(optimpar)

        x  = np.asarray(x, dtype=float)
        nr, nc = x.shape

        # Remove all-NA rows before fitting (R's vsnSample does this)
        all_na  = np.all(~np.isfinite(x), axis=1)
        n_allna = int(all_na.sum())
        if n_allna > 0:
            warnings.warn(f"{n_allna} rows removed (all NA) before fitting.")
        x_fit = x[~all_na, :]

        if verbose:
            print(f"vsn2: {nr} x {nc} matrix (1 stratum).")

        # LTS ML fitting
        p_fit = _vsn_lts(x_fit, calib, lts_quantile, opar)

        # Extract parameters
        if calib == "affine":
            a_fit    = p_fit[:nc]
            logb_fit = p_fit[nc:]
        else:
            a_fit    = np.zeros(nc)
            logb_fit = p_fit
        b_fit = np.exp(logb_fit)

        # hoffset = log2(2 * geomean(b_j))  (R: log2(2*scalingFactorTransformation(mean(log_b))))
        # scalingFactorTransformation(b) = exp(b)  so geomean = exp(mean(log_b)) = exp(mean(logb_fit))
        hoffset = np.log2(2.0 * np.exp(float(np.mean(logb_fit))))

        # Apply to ALL data: hx = arcsinh(b*y + a) / log(2) - hoffset
        hx = np.full_like(x, np.nan)
        for j in range(nc):
            valid = np.isfinite(x[:, j])
            if valid.any():
                u = b_fit[j] * x[valid, j] + a_fit[j]
                # BUG 1 FIX: divide by log(2) to convert arcsinh → log2 scale
                hx[valid, j] = np.arcsinh(u) / np.log(2.0) - hoffset

        if verbose:
            print("Please use a mean-SD plot to verify the fit.")

        return hx

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
        with mo.status.spinner(subtitle="Running VSN Normalization…"):
            _mat = df_raw[cols_detected].values.copy().astype(float)
            _mat[_mat <= 0] = np.nan

            # vsn2() now returns the correctly scaled log2 matrix directly
            normalized_mat = vsn2(_mat, calib=calib_type.value, verbose=True)

            _meta_cols = [c for c in df_raw.columns if c not in cols_detected]
            df_meta = df_raw[_meta_cols].copy()
            df_norm = pd.DataFrame(
                normalized_mat,
                columns=[f"vsn_{c}" for c in cols_detected],
                index=df_raw.index,
            )
            df_pow2 = pd.DataFrame(
                2.0 ** normalized_mat,
                columns=[f"pow2_{c}" for c in cols_detected],
                index=df_raw.index,
            )
            df_out = pd.concat([df_meta, df_raw[cols_detected], df_norm, df_pow2], axis=1)

    return df_out, normalized_mat


@app.cell(hide_code=True)
def _(calib_type, cols_detected, df_out, df_raw, file_input,
      intensity_prefix, io, linregress, mo, normalized_mat, np, plt,
      run_btn, sample_select):

    # --- Sidebar ---
    _sidebar = mo.vstack([
        mo.md("### Options"),
        file_info,      # This references the new mo.md block containing the links
        file_input,     # This references the updated mo.ui.file
        intensity_prefix,
        calib_type,
        mo.md("---"),
        run_btn,
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
        # Mimics R's meanSdPlot(ranks=TRUE) + histogram + boxplot
        _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.5))

        _row_means = np.nanmean(normalized_mat, axis=1)
        _row_sds   = np.nanstd(normalized_mat, axis=1)
        _valid     = np.isfinite(_row_means) & np.isfinite(_row_sds)

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

        # ── Right plot: boxplot per sample ────────────────────────────────────
        _bp_data = [normalized_mat[:, j][np.isfinite(normalized_mat[:, j])]
                    for j in range(normalized_mat.shape[1])]
        _axes[2].boxplot(_bp_data, patch_artist=True,
                         boxprops=dict(facecolor="lightsteelblue"),
                         flierprops=dict(marker=".", markersize=1, alpha=0.3),
                         showfliers=False)
        _axes[2].set_title("VSN Normalized — per sample")
        _axes[2].set_ylabel("VSN (log2 scale)")
        _short_names = [c.replace(intensity_prefix.value, "").lstrip("_. ") or f"S{i+1}"
                        for i, c in enumerate(cols_detected)]
        _axes[2].set_xticks(range(1, len(_short_names) + 1))
        _axes[2].set_xticklabels(_short_names, rotation=45, ha="right", fontsize=7)

        plt.tight_layout()
        _diagnostics_ui = mo.as_html(_fig)
        plt.close(_fig)

        # ── Regression tab ────────────────────────────────────────────────────
        if sample_select is not None and sample_select.value:
            _sel = sample_select.value
            _raw_v = df_raw[_sel].values.copy().astype(float)
            _raw_v[_raw_v <= 0] = np.nan
            _vsn_v = df_out[f"vsn_{_sel}"].values
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
                _ax_r.set_ylabel(f"vsn_{_sel}")
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

        # ── Download button ──────────────────────────────────────────────────
        _tsv_buf = io.BytesIO()
        df_out.to_csv(_tsv_buf, index=False, sep="\t")
        _dl = mo.download(
            data=_tsv_buf.getvalue(),
            filename="normalized_vsn_data.txt",
            label="📥 Download VSN Normalized Data",
            mimetype="text/tab-separated-values",
        )

        _tabs = mo.ui.tabs({
            "Preview":     _preview_ui,
            "Diagnostics": _diagnostics_ui,
            "Regression":  _regression_ui,
        })

        _main = mo.vstack([
            mo.hstack(
                [mo.md(f"✅ Successfully normalized **{len(cols_detected)}** columns."), _dl],
                justify="space-between", align="center",
            ),
            mo.md("---"),
            _tabs,
        ])

    dashboard = mo.vstack([
        mo.md("# Variance Stabilizing Normalization (VSN) Dashboard"),
        mo.hstack([_sidebar, _main], widths=[1, 3], gap=2),
    ])
    dashboard
    return dashboard,


if __name__ == "__main__":
    app.run()