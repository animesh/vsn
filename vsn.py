"""
vsn2.py — Pure Python/NumPy port of the VSN (Variance Stabilization and Normalization)
algorithm originally implemented in R/C by Wolfgang Huber et al.

Reference:
  Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. (2002)
  Variance stabilization applied to microarray data calibration and to the
  quantification of differential expression.  Bioinformatics 18(suppl 1), S96-S104.

All dependencies are standard Python scientific stack: numpy, scipy.
No R or compiled C code is required.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Constants
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

# ---------------------------------------------------------------------------
# Data classes (replacing R S4 classes vsn and vsnInput)
# ---------------------------------------------------------------------------


@dataclass
class VsnResult:
    """Stores the result of a VSN fit (equivalent to R's 'vsn' S4 class).

    Attributes
    ----------
    coefficients : np.ndarray, shape (n_strata, n_cols_d2, 2)
        Fitted offset ([:,:,0]) and log-scale ([:,:,1]) parameters.
    strata : np.ndarray of int, shape (n_rows,) or empty
        Integer stratum label for each row (1-based).
    mu : np.ndarray, shape (n_rows,)
        Row means in transformed space (profile-likelihood estimate).
    sigsq : float
        Estimated residual variance.
    hx : np.ndarray or None
        Transformed data matrix (set when returnData=True).
    lbfgsb : int
        Optimizer return code (0 = success).
    hoffset : np.ndarray
        log2-scale offset applied after transformation.
    calib : str
        Calibration mode: 'affine' or 'none'.
    """

    coefficients: np.ndarray
    strata: np.ndarray
    mu: np.ndarray
    sigsq: float
    hx: Optional[np.ndarray]
    lbfgsb: int
    hoffset: np.ndarray
    calib: str

    def nrow(self) -> int:
        return len(self.mu)


@dataclass
class VsnInput:
    """Input data structure for the VSN algorithm (equivalent to R's 'vsnInput').

    Attributes
    ----------
    x : np.ndarray, shape (n_rows, n_cols)
        Data matrix.
    reference : VsnResult or None
        If provided, normalize against this reference.
    strata : np.ndarray of int, shape (n_rows,) or empty
        Integer stratum labels (1-based). Empty array means one stratum.
    ordered : bool
        Whether rows are already sorted by stratum.
    lts_quantile : float
        LTS quantile (fraction of data to use per iteration).
    subsample : int
        Number of rows to subsample (0 = use all).
    verbose : bool
    calib : str
        'affine' or 'none'.
    pstart : np.ndarray, shape (n_strata, d2, 2)
        Starting parameters.
    optimpar : dict
        Optimization parameters (keys matching DEFAULT_OPTIMPAR).
    """

    x: np.ndarray
    reference: Optional[VsnResult]
    strata: np.ndarray
    ordered: bool
    lts_quantile: float
    subsample: int
    verbose: bool
    calib: str
    pstart: np.ndarray
    optimpar: dict

    def nrow(self) -> int:
        return self.x.shape[0]

    def ncol(self) -> int:
        return self.x.shape[1]

    def n_strata(self) -> int:
        """Number of unique strata."""
        return int(np.max(self.strata)) if len(self.strata) > 0 else 1

    def subset_rows(self, idx: np.ndarray) -> "VsnInput":
        """Return a copy with rows restricted to idx (0-based)."""
        new_strata = self.strata[idx] if len(self.strata) > 0 else self.strata
        new_ref_mu: Optional[np.ndarray] = None
        new_ref_sigsq: float = np.nan
        new_ref = None
        if self.reference is not None and len(self.reference.mu) > 0:
            new_ref_mu = self.reference.mu[idx]
            new_ref_sigsq = self.reference.sigsq
            new_ref = VsnResult(
                coefficients=self.reference.coefficients,
                strata=self.reference.strata,
                mu=new_ref_mu,
                sigsq=new_ref_sigsq,
                hx=None,
                lbfgsb=0,
                hoffset=self.reference.hoffset,
                calib=self.reference.calib,
            )
        return VsnInput(
            x=self.x[idx, :],
            reference=new_ref,
            strata=new_strata,
            ordered=False,
            lts_quantile=self.lts_quantile,
            subsample=self.subsample,
            verbose=self.verbose,
            calib=self.calib,
            pstart=self.pstart,
            optimpar=self.optimpar,
        )

    def subset_cols(self, col_idx) -> "VsnInput":
        """Return a copy with columns restricted to col_idx."""
        return VsnInput(
            x=self.x[:, col_idx : col_idx + 1],
            reference=self.reference,
            strata=self.strata,
            ordered=self.ordered,
            lts_quantile=self.lts_quantile,
            subsample=self.subsample,
            verbose=self.verbose,
            calib=self.calib,
            pstart=self.pstart[:, col_idx : col_idx + 1, :],
            optimpar=self.optimpar,
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _calib_to_int(calib: str) -> int:
    """Map calibration string to integer code used internally (mirrors calibCharToInt)."""
    if calib == "affine":
        return 0
    if calib == "none":
        return 1
    raise ValueError(f"Invalid value of 'calib': {calib!r}")


def is_small(x: np.ndarray, tol: float = np.sqrt(np.finfo(float).eps)) -> bool:
    """Return True if all absolute values of x are below tol."""
    return bool(np.max(np.abs(x)) < tol)


def row_variances(x: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
    """Row-wise sample variances, ignoring NaN."""
    n = np.sum(~np.isnan(x), axis=1).astype(float)
    n[n < 1] = np.nan
    if mean is None:
        mean = np.nanmean(x, axis=1)
    diff = x - mean[:, np.newaxis]
    return np.nansum(diff**2, axis=1) / (n - 1)


def scaling_factor_transformation(b: np.ndarray) -> np.ndarray:
    """f(b) = exp(b)  — the 'f' function from the VSN vignette."""
    return np.exp(b)


# ---------------------------------------------------------------------------
# Core transformation: apply asinh(exp(b)*y + a)
# ---------------------------------------------------------------------------


def _calc_trsf(y: np.ndarray, par: np.ndarray, strat: np.ndarray, calib_int: int) -> np.ndarray:
    """Apply the VSN transformation to y.

    Parameters
    ----------
    y : (nr, nc) float array
    par : flat parameter vector
        For 'affine' (calib_int==0): length = ns*nc*2  (offsets then log-scales)
        For 'none'   (calib_int!=0): length = 2        [a, log_b]
    strat : (nr,) int array, 1-based stratum indices (used only for affine)
    calib_int : 0 for affine, anything else for none

    Returns
    -------
    hy : (nr, nc) float array
    """
    nr, nc = y.shape
    hy = np.full_like(y, np.nan)

    if calib_int == 0:
        # affine calibration: one (a, b) per stratum per column
        ns = par.shape[0] // (nc * 2)
        # par layout: [a_{s,j}] then [log_b_{s,j}]  flattened col-major
        # par[s + j*ns]        = a_{s+1, j+1}
        # par[s + j*ns + ns*nc] = log_b_{s+1, j+1}
        for j in range(nc):
            for i in range(nr):
                z = y[i, j]
                if np.isnan(z):
                    continue
                s = int(strat[i]) - 1  # 0-based stratum index
                a = par[s + j * ns]
                log_b = par[s + j * ns + ns * nc]
                fb = np.exp(log_b)
                hy[i, j] = np.arcsinh(fb * z + a)
    else:
        # no calibration: single (a, log_b) pair
        a = par[0]
        fb = np.exp(par[1])
        mask = ~np.isnan(y)
        hy[mask] = np.arcsinh(fb * y[mask] + a)

    return hy


# ---------------------------------------------------------------------------
# Log-likelihood and gradient (pure Python/NumPy, mirroring vsn2.c)
# ---------------------------------------------------------------------------


def _negloglik_and_grad(
    par: np.ndarray,
    y: np.ndarray,
    istrat: np.ndarray,
    mu_fixed: Optional[np.ndarray],
    sigsq_fixed: Optional[float],
    profiling: bool,
) -> tuple[float, np.ndarray]:
    """Compute negative log-likelihood and gradient for VSN.

    Parameters
    ----------
    par : (2*ns,) array  — first ns are offsets a_j, next ns are log-scales b_j
    y : (nr, nc) data matrix (column-major order mimicked; rows=features, cols=samples)
    istrat : (ns+1,) integer array — C-style start indices into flattened y (column-major)
    mu_fixed : (nr,) array or None — if provided, use as fixed mu (no profiling)
    sigsq_fixed : float or None — fixed sigsq (only if not profiling)
    profiling : bool — True = profile likelihood (estimate mu, sigsq from data)

    Returns
    -------
    (ll, grad) where ll is the scalar negative log-likelihood and grad has shape (2*ns,)
    """
    nr, nc = y.shape
    ns = len(istrat) - 1
    ntot = int(np.sum(~np.isnan(y)))

    a = par[:ns]
    b = par[ns:]

    # Flatten y in column-major order (Fortran order) to match the C indexing
    y_flat = y.flatten(order="F")  # index [i + j*nr]

    # --- 1st sweep: compute Y_ki = b_j*y_ki + a_j,  h_ki = asinh(Y_ki),  A_ki ---
    ly = np.full(nr * nc, np.nan)
    asly = np.full(nr * nc, np.nan)
    ma = np.full(nr * nc, np.nan)

    jac1 = 0.0
    jac2 = 0.0

    for j in range(ns):
        aj = a[j]
        bj = np.exp(b[j])
        start = istrat[j]
        end = istrat[j + 1]
        chunk = y_flat[start:end]
        valid = ~np.isnan(chunk)
        z = bj * chunk[valid] + aj
        idx_valid = np.where(valid)[0] + start
        ly[idx_valid] = z
        asly[idx_valid] = np.arcsinh(z)
        ma[idx_valid] = 1.0 / np.sqrt(1.0 + z**2)
        jac1 += np.sum(np.log(1.0 + z**2))
        jac2 += np.sum(valid) * np.log(bj)

    jacobian = 0.5 * jac1 - jac2

    # Reshape asly back to matrix (column-major → matrix)
    asly_mat = asly.reshape((nr, nc), order="F")

    # --- 2nd sweep: compute residuals r_ki ---
    if profiling:
        mu = np.nanmean(asly_mat, axis=1)  # shape (nr,)
    else:
        mu = mu_fixed  # type: ignore[assignment]

    resid_mat = asly_mat - mu[:, np.newaxis]  # broadcast over cols
    # NaN where either mu or asly is NaN
    nan_mask = np.isnan(asly_mat) | np.isnan(mu[:, np.newaxis])
    resid_mat[nan_mask] = np.nan
    resid_flat = resid_mat.flatten(order="F")

    ssq = np.nansum(resid_mat**2)
    nt = float(ntot)

    if profiling:
        sigsq = ssq / nt
        residuals = nt / 2.0
    else:
        sigsq = sigsq_fixed  # type: ignore[assignment]
        residuals = ssq / (2.0 * sigsq)

    scale = nt / 2.0 * np.log(2.0 * np.pi * sigsq)
    ll = scale + residuals + jacobian

    # --- Gradient ---
    grad = np.zeros(2 * ns)
    rfac = 1.0 / sigsq

    for j in range(ns):
        start = istrat[j]
        end = istrat[j + 1]
        bj = np.exp(b[j])
        sa = 0.0
        sb = 0.0
        nj = 0
        chunk_r = resid_flat[start:end]
        chunk_ma = ma[start:end]
        chunk_y = y_flat[start:end]
        chunk_ly = ly[start:end]

        valid = ~np.isnan(chunk_r)
        if np.any(valid):
            rv = chunk_r[valid]
            mv = chunk_ma[valid]
            yv = chunk_y[valid]
            lyv = chunk_ly[valid]
            nj = int(np.sum(valid))
            z = rv * rfac + mv * lyv
            sa = float(np.sum(z * mv))
            sb = float(np.sum(z * mv * yv))

        grad[j] = sa
        # DFDB(b[j]) = exp(b[j])
        grad[j + ns] = np.exp(b[j]) * (sb - nj / bj)

    return ll, grad


# ---------------------------------------------------------------------------
# calcistrat: compute strat index array (C-style, column-major)
# ---------------------------------------------------------------------------


def _calc_istrat(v: VsnInput) -> np.ndarray:
    """Compute istrat: start positions of each stratum in column-major flattened y.

    For 'affine': istrat has length (ncol * n_strata + 1).
    For 'none': istrat = [0, nrow*ncol].
    """
    nr = v.x.shape[0]
    nc = v.x.shape[1]
    nrs = v.n_strata()
    calib = v.calib

    if calib == "affine":
        strata = v.strata  # 1-based
        if len(strata) > 0:
            assert v.ordered, "Rows must be ordered by stratum before calcistrat (affine)."
            # positions where stratum changes (0-based row index)
            # equivalent to which(!duplicated(v@strata)) - 1 in R
            istr = np.where(np.concatenate(([True], np.diff(strata) != 0)))[0]  # 0-based
        else:
            istr = np.array([0], dtype=np.intp)

        assert len(istr) == nrs, f"Expected {nrs} strata starts, got {len(istr)}"

        istrat = np.zeros(nc * nrs + 1, dtype=np.intp)
        for i in range(nc):
            istrat[i * nrs: i * nrs + nrs] = i * nr + istr
        istrat[-1] = nr * nc
        return istrat

    elif calib == "none":
        if nrs > 1:
            raise ValueError("There must not be more than 1 stratum if 'calib' is 'none'.")
        return np.array([0, nr * nc], dtype=np.intp)

    else:
        raise ValueError(f"Invalid value of 'calib': {calib!r}")


# ---------------------------------------------------------------------------
# vsnML: maximum-likelihood estimator (replaces R vsnML + C vsn2_optim)
# ---------------------------------------------------------------------------


def vsn_ml(v: VsnInput) -> VsnResult:
    """Fit VSN parameters by maximum likelihood using L-BFGS-B.

    Corresponds to R's vsnML() which calls the C vsn2_optim routine.

    Parameter layout (flat vector passed to optimizer):
        par[:ns]  = offsets a_{s,j}     (column-major over strata×samples)
        par[ns:]  = log-scales b_{s,j}
    where ns = n_strata * n_cols for 'affine', or 1 for 'none'.
    """
    istrat = _calc_istrat(v)
    ns = len(istrat) - 1

    nr, nc = v.x.shape
    nrs = v.n_strata()

    # Build flat parameter vector in column-major (stratum-major) order matching C
    # pstart shape: (n_strata, d2, 2)  [:,:,0]=offsets, [:,:,1]=log-scales
    if v.calib == "affine":
        a_init = v.pstart[:, :, 0].flatten(order="F")  # (nrs*nc,)
        b_init = v.pstart[:, :, 1].flatten(order="F")
    else:
        a_init = v.pstart[:, :, 0].flatten()
        b_init = v.pstart[:, :, 1].flatten()

    p0 = np.concatenate([a_init, b_init])  # length 2*ns

    # Reference: mu and sigsq
    ref_mu: Optional[np.ndarray] = None
    ref_sigsq: Optional[float] = None
    profiling = True

    if v.reference is not None and len(v.reference.mu) > 0:
        ref_mu = v.reference.mu
        ref_sigsq = float(v.reference.sigsq)
        profiling = False

    opar = v.optimpar

    # Bounds: offsets are unbounded; log-scale parameters are in [-100, 100]
    bounds = [(None, None)] * ns + [(-100.0, 100.0)] * ns

    # Cache for mu/sigsq (computed as side-effect of each likelihood evaluation)
    sigsq_state: list = [np.nan]
    mu_state: list = [None]

    def _negloglik(par):
        ll, g = _negloglik_and_grad(par, v.x, istrat, ref_mu, ref_sigsq, profiling)
        if profiling:
            # Recompute mu and sigsq at this parameter value for caching.
            # This mirrors the internal computation in _negloglik_and_grad.
            a_ = par[:ns]
            b_ = np.exp(par[ns:])
            y_flat = v.x.flatten(order="F")
            asly = np.full(nr * nc, np.nan)
            for jj in range(ns):
                s, e = istrat[jj], istrat[jj + 1]
                chunk = y_flat[s:e]
                valid = ~np.isnan(chunk)
                if np.any(valid):
                    z = b_[jj] * chunk[valid] + a_[jj]
                    idx_v = np.where(valid)[0] + s
                    asly[idx_v] = np.arcsinh(z)
            asly_mat = asly.reshape((nr, nc), order="F")
            _mu = np.nanmean(asly_mat, axis=1)
            resid = asly_mat - _mu[:, np.newaxis]
            ntot = int(np.sum(~np.isnan(v.x)))
            sigsq_state[0] = float(np.nansum(resid**2) / ntot)
            mu_state[0] = _mu
        return ll, g

    result = minimize(
        _negloglik,
        p0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": opar["maxit"],
            "maxcor": 5,
            "ftol": opar["factr"] * np.finfo(float).eps,
            "gtol": opar["pgtol"],
        },
    )

    fail = 0 if result.success else result.status

    # Run one final evaluation at the optimum to populate mu/sigsq caches
    _negloglik(result.x)
    final_sigsq = sigsq_state[0] if profiling else ref_sigsq  # type: ignore[assignment]
    final_mu = mu_state[0] if profiling else ref_mu  # type: ignore[assignment]

    # Reshape coefficients back to (nrs, d2, 2)
    opt_par = result.x
    d2 = nc if v.calib == "affine" else 1
    a_opt = opt_par[:ns].reshape((nrs, d2), order="F")
    b_opt = opt_par[ns:].reshape((nrs, d2), order="F")
    coefficients = np.stack([a_opt, b_opt], axis=-1)  # (nrs, d2, 2)

    hoffset = np.full(nrs, np.nan)

    if fail != 0 and opar.get("trace", 0) >= 0:
        nrp = min(len(p0), 6)
        warnings.warn(
            f"L-BFGS-B optimizer returned fail={fail}. "
            f"pstart[:{nrp}]={np.round(p0[:nrp], 4)}, "
            f"coef[:{nrp}]={np.round(opt_par[:nrp], 4)}"
        )

    if final_mu is None:
        final_mu = np.full(nr, np.nan)

    return VsnResult(
        coefficients=coefficients,
        strata=v.strata,
        mu=final_mu,
        sigsq=float(final_sigsq),
        hx=np.full((nr, 0), np.nan),
        lbfgsb=fail,
        hoffset=hoffset,
        calib=v.calib,
    )


# ---------------------------------------------------------------------------
# vsnLTS: LTS robust wrapper around vsn_ml
# ---------------------------------------------------------------------------


def vsn_lts(v: VsnInput) -> VsnResult:
    """Least-trimmed-squares robust modification of the ML estimator.

    Corresponds to R's vsnLTS().
    """
    old_hy = np.full_like(v.x, np.inf)
    cvgc_cnt = 0

    intstrata = v.strata if len(v.strata) > 0 else np.ones(v.nrow(), dtype=int)

    whsel: Optional[np.ndarray] = None  # selected row indices (0-based)

    rsv: Optional[VsnResult] = None

    for iter_idx in range(1, v.optimpar["cvg_niter"] + 1):
        sv = v if (iter_idx == 1 or whsel is None) else v.subset_rows(whsel)
        rsv = vsn_ml(sv)

        # If LTS quantile == 1, no trimming needed
        if is_small(v.lts_quantile - 1.0):
            break

        # Apply to all data
        hy = vsn2_trsf(
            x=v.x,
            p=rsv.coefficients,
            strata=intstrata,
            calib=v.calib,
        )
        # Update pstart
        v = _replace_pstart(v, rsv.coefficients)

        # Residuals
        if v.reference is not None and len(v.reference.mu) > 0:
            hmean = v.reference.mu
        else:
            hmean = np.nanmean(hy, axis=1)
            if iter_idx == 1:
                if rsv.lbfgsb == 0:
                    assert is_small(rsv.mu - hmean), "mu mismatch at iter 1"
            else:
                if rsv.lbfgsb == 0 and whsel is not None:
                    assert is_small(rsv.mu - hmean[whsel]), "mu mismatch"
                tmp = np.full(v.nrow(), np.nan)
                if whsel is not None:
                    tmp[whsel] = rsv.mu
                rsv.mu = tmp

        squared_residuals = (hy - hmean[:, np.newaxis]) ** 2
        rvar = np.sum(squared_residuals, axis=1)

        # Select rows within quantile, per stratum × intensity-slice
        n_slices = 5
        # Reproduce R's rank(hmean, na.last=TRUE): 1-based, average ties, NaN last
        rank_hmean = _rank_na_last(hmean)
        rank_min = float(np.min(rank_hmean))
        rank_max = float(np.max(rank_hmean))
        rank_span = rank_max - rank_min

        # Match R's cut(rank_hmean, breaks=n_slices) exactly. When breaks is
        # a scalar, R first creates equally spaced breaks over [min, max] and
        # then expands only the two outer endpoints by 0.1% of the range.
        # Expanding the range before linspace shifts every internal boundary,
        # which changes the sparse-data LTS selection.
        cut_breaks = np.linspace(rank_min, rank_max, n_slices + 1)
        cut_breaks[0] -= rank_span * 0.001
        cut_breaks[-1] += rank_span * 0.001
        slice_labels = np.searchsorted(cut_breaks, rank_hmean, side="left") - 1
        slice_labels = np.clip(slice_labels, 0, n_slices - 1)

        nrs = int(np.max(intstrata))
        whsel_mask = np.zeros(v.nrow(), dtype=bool)

        for sl in range(n_slices):
            for st in range(1, nrs + 1):
                mask = (slice_labels == sl) & (intstrata == st)
                if not np.any(mask):
                    continue
                q = np.nanquantile(rvar[mask], v.lts_quantile)
                # Include if rvar <= quantile OR in lowest-intensity slice (sl==0)
                keep = mask & ((rvar <= q) | (slice_labels == 0))
                whsel_mask |= keep

        whsel = np.where(whsel_mask)[0]

        # Convergence check
        if v.optimpar["cvg_eps"] > 0:
            cvgc = np.nanmax(np.abs(hy - old_hy))
            cvgc_cnt = (cvgc_cnt + 1) if cvgc < v.optimpar["cvg_eps"] else 0
            if cvgc_cnt >= 3:
                break
            old_hy = hy

    assert rsv is not None
    return rsv


def _replace_pstart(v: VsnInput, coef: np.ndarray) -> VsnInput:
    """Return a copy of VsnInput with pstart replaced by coef."""
    import dataclasses
    return dataclasses.replace(v, pstart=coef)


def _rank_na_last(x: np.ndarray) -> np.ndarray:
    """Reproduce R's rank(x, na.last=TRUE) with average ties.

    Non-NaN values are ranked 1..n_valid using average for ties.
    NaN values are assigned sequential ranks n_valid+1, n_valid+2, ...
    in their order of appearance, matching R's na.last=TRUE behaviour.
    """
    n = len(x)
    nan_mask = np.isnan(x)
    n_valid = int(np.sum(~nan_mask))

    out = np.empty(n)
    valid_vals = x[~nan_mask]
    # scipy rankdata uses average ties and 1-based ranks — same as R default
    out[~nan_mask] = rankdata(valid_vals, method="average")

    # NaNs get successive ranks beyond the valid range, in order of appearance
    for k, idx in enumerate(np.where(nan_mask)[0]):
        out[idx] = n_valid + 1 + k

    return out


# ---------------------------------------------------------------------------
# vsn2trsf: apply transformation to a matrix
# ---------------------------------------------------------------------------


def vsn2_trsf(
    x: np.ndarray,
    p: np.ndarray,
    strata: np.ndarray,
    hoffset: Optional[np.ndarray] = None,
    calib: str = "affine",
) -> np.ndarray:
    """Apply the VSN glog transformation to x.

    Parameters
    ----------
    x : (nr, nc) float matrix
    p : (n_strata, d2, 2) coefficient array  [:,:,0]=offset, [:,:,1]=log-scale
    strata : (nr,) int array (1-based) or empty
    hoffset : (n_strata,) array or None
    calib : 'affine' or 'none'

    Returns
    -------
    hx : (nr, nc) transformed matrix
        If hoffset is provided, values are in log2 scale (standard VSN output).
        Otherwise raw asinh values.
    """
    nr, nc = x.shape

    if len(strata) == 0:
        strata = np.ones(nr, dtype=int)
    nrstrata = int(np.max(strata))

    if p.ndim == 2:
        p = p[np.newaxis, :, :]  # add stratum dimension

    calib_int = _calib_to_int(calib)

    # Build flat par vector in the C convention (col-major within strata)
    nrs = p.shape[0]
    d2 = p.shape[1]  # nc for affine, 1 for none

    if calib_int == 0:
        # affine: par[s + j*nrs] = a_{s,j}, par[s + j*nrs + nrs*nc] = log_b_{s,j}
        a_flat = p[:, :, 0].flatten(order="F")  # shape (nrs*nc,)
        b_flat = p[:, :, 1].flatten(order="F")
        par = np.concatenate([a_flat, b_flat])
    else:
        par = np.array([p[0, 0, 0], p[0, 0, 1]])

    hx = _calc_trsf(x, par, strata, calib_int)

    if hoffset is not None:
        assert len(hoffset) == nrstrata
        hx = hx / np.log(2) - hoffset[strata - 1][:, np.newaxis]  # strata is 1-based

    return hx


# ---------------------------------------------------------------------------
# pstartHeuristic: initial parameters
# ---------------------------------------------------------------------------


def pstart_heuristic(x: np.ndarray, sp: dict, calib: str) -> np.ndarray:
    """Match R pstartHeuristic: offsets=0 and log-scale parameters=1."""
    d2 = x.shape[1] if calib == "affine" else 1
    pstart = np.zeros((len(sp), d2, 2), dtype=float)
    pstart[:, :, 1] = 1.0
    return pstart


# ---------------------------------------------------------------------------
# int2strata: integer array to 1-based stratum labels
# ---------------------------------------------------------------------------


def int_to_strata(strata: np.ndarray) -> np.ndarray:
    """Validate and return integer stratum array (1-based).

    Equivalent to R's int2factor with integer checking.
    """
    strata = np.asarray(strata, dtype=int)
    ssu = np.unique(strata)
    expected = np.arange(1, len(ssu) + 1)
    if not np.array_equal(ssu, expected):
        raise ValueError(
            "'strata' must be an integer vector with values covering 1..n "
            f"(got unique values: {ssu})"
        )
    return strata


# ---------------------------------------------------------------------------
# vsnColumnByColumn
# ---------------------------------------------------------------------------


def vsn_column_by_column(v: VsnInput) -> VsnResult:
    """Fit each column independently against its own reference. Used when a
    reference is provided (normalise to reference, column by column).

    Corresponds to R's vsnColumnByColumn().
    """
    nc = v.ncol()
    nrs = v.n_strata()

    first_col = v.subset_cols(0)
    rlv = vsn_lts(first_col)

    d = rlv.coefficients.shape
    assert d[1] == 1, "Expected single-column coefficient array."

    cf = np.full((d[0], nc, d[2]), np.nan)
    cf[:, 0, :] = rlv.coefficients[:, 0, :]

    for j in range(1, nc):
        cf[:, j, :] = vsn_lts(v.subset_cols(j)).coefficients[:, 0, :]

    ref_mu = v.reference.mu if (v.reference is not None and len(v.reference.mu) > 0) else np.array([])
    ref_sigsq = v.reference.sigsq if v.reference is not None else np.nan

    return VsnResult(
        coefficients=cf,
        strata=v.strata,
        mu=ref_mu,
        sigsq=float(ref_sigsq),
        hx=None,
        lbfgsb=0,
        hoffset=np.full(nrs, np.nan),
        calib=v.calib,
    )


# ---------------------------------------------------------------------------
# vsnStrata
# ---------------------------------------------------------------------------


def vsn_strata(v: VsnInput) -> VsnResult:
    """Reorder rows by stratum if necessary, fit, then undo reordering.

    Corresponds to R's vsnStrata().
    """
    ord_idx: Optional[np.ndarray] = None
    nrs = v.n_strata()

    if nrs > 1:
        ord_idx = np.argsort(v.strata, kind="stable")
        v = v.subset_rows(ord_idx)

    import dataclasses
    v = dataclasses.replace(v, ordered=True)

    has_ref = v.reference is not None and len(v.reference.mu) > 0
    res = vsn_column_by_column(v) if has_ref else vsn_lts(v)

    # Undo the row ordering in mu
    if ord_idx is not None:
        full_mu = np.empty_like(res.mu)
        full_mu[ord_idx] = res.mu
        res.mu = full_mu

    return res


# ---------------------------------------------------------------------------
# vsnSample
# ---------------------------------------------------------------------------


def vsn_sample(v: VsnInput) -> VsnResult:
    """Handle subsampling, NA rows, and delegate to vsn_strata.

    Corresponds to R's vsnSample().
    """
    has_ref = v.reference is not None and len(v.reference.mu) > 0

    if v.subsample > 0 and has_ref:
        raise ValueError(
            "The 'subsample' and 'normalization to reference' options cannot be mixed."
        )

    wh: Optional[np.ndarray] = None

    if v.subsample > 0:
        nrs = v.n_strata()
        if len(v.strata) > 0 and nrs > 1:
            wh_list = []
            for st in range(1, nrs + 1):
                idx = np.where(v.strata == st)[0]
                chosen = np.random.choice(idx, size=min(v.subsample, len(idx)), replace=False)
                wh_list.append(chosen)
            wh = np.concatenate(wh_list)
        else:
            wh = np.random.choice(v.nrow(), size=min(v.subsample, v.nrow()), replace=False)

    if has_ref and v.reference is not None:
        wh_ref = np.where(~np.isnan(v.reference.mu))[0]
        wh = wh_ref

    # Remove all-NA rows
    all_na = np.all(np.isnan(v.x), axis=1)
    num_na = int(np.sum(all_na))
    if num_na > 0:
        warnings.warn(f"{num_na} rows were removed since they contained only NA elements.")
        if wh is None:
            wh = np.where(~all_na)[0]
        else:
            wh = np.setdiff1d(wh, np.where(all_na)[0])

    if wh is not None:
        res = vsn_strata(v.subset_rows(wh))
        new_mu = np.full(v.nrow(), np.nan)
        new_mu[wh] = res.mu
        res.mu = new_mu
    else:
        res = vsn_strata(v)

    return res


# ---------------------------------------------------------------------------
# vsnMatrix: main entry point
# ---------------------------------------------------------------------------


def vsn_matrix(
    x: np.ndarray,
    reference: Optional[VsnResult] = None,
    strata: Optional[np.ndarray] = None,
    lts_quantile: float = 0.9,
    subsample: int = 0,
    verbose: bool = False,
    return_data: bool = True,
    calib: str = "affine",
    pstart: Optional[np.ndarray] = None,
    min_data_points_per_stratum: int = 42,
    optimpar: Optional[dict] = None,
    defaultpar: Optional[dict] = None,
) -> VsnResult:
    """Fit the VSN transformation to a data matrix.

    Parameters
    ----------
    x : (nr, nc) float array  — expression matrix (rows=features, cols=samples)
    reference : VsnResult or None
        If provided, normalize against this fitted reference.
    strata : (nr,) int array or None
        Integer stratum labels (1-based, covering 1..n_strata).
    lts_quantile : float
        LTS robustness quantile.
    subsample : int
        If > 0, use only this many rows per stratum for fitting.
    verbose : bool
    return_data : bool
        If True, compute and store the transformed data matrix in the result.
    calib : str
        'affine' (per-sample calibration) or 'none' (single global transform).
    pstart : array or None
        Starting parameters; if None, computed by pstart_heuristic.
    min_data_points_per_stratum : int
        Raise an error if fewer than this many rows per stratum.
    optimpar : dict or None
        Override specific optimization parameters (keys from OPTIMPAR_NAMES).
    defaultpar : dict or None
        Full set of default optimization parameters.

    Returns
    -------
    VsnResult with fitted parameters and (optionally) transformed data.
    """
    x = np.asarray(x, dtype=float)
    subsample = int(subsample)
    nr, nc = x.shape

    # ---- strata ----
    if strata is None:
        strata_arr = np.array([], dtype=int)
        strata_split = {"all": np.arange(nr)}
    else:
        strata_arr = int_to_strata(np.asarray(strata, dtype=int))
        nrs = int(np.max(strata_arr))
        strata_split = {str(s): np.where(strata_arr == s)[0] for s in range(1, nrs + 1)}

    # ---- check stratum sizes ----
    for label, idx in strata_split.items():
        if len(idx) < min_data_points_per_stratum:
            n_strata_total = len(strata_split)
            if n_strata_total > 1:
                sizes = sorted({len(v) for v in strata_split.values()})
                raise ValueError(
                    f"Some strata have too few rows for reliable estimation. "
                    f"Stratum sizes: {sizes}. Consider reducing the number of strata, "
                    f"or set min_data_points_per_stratum to a lower value."
                )
            else:
                raise ValueError(
                    f"The number of rows in the data matrix, {nr}, is too small for "
                    f"reliable estimation of the vsn transformation parameters. "
                    f"Set min_data_points_per_stratum to a lower value if you are sure."
                )

    # ---- reference ----
    if reference is None:
        if nc <= 1:
            raise ValueError("'x' needs to have 2 or more columns if no 'reference' is specified.")
        ref = None
    else:
        if reference.nrow() != nr:
            raise ValueError("'nrow(reference)' must be equal to 'nrow(x)'.")
        if len(reference.mu) != nr:
            raise ValueError(
                f"reference.mu has length {len(reference.mu)}, expected {nr}."
            )
        ref = reference

    # ---- optimpar ----
    if defaultpar is None:
        defaultpar = dict(DEFAULT_OPTIMPAR)
    else:
        defaultpar = dict(DEFAULT_OPTIMPAR, **defaultpar)

    if optimpar is not None:
        bad = set(optimpar) - OPTIMPAR_NAMES
        if bad:
            raise ValueError(f"Unknown optimpar keys: {bad}. Valid keys: {OPTIMPAR_NAMES}")
        defaultpar.update(optimpar)
    opar = defaultpar

    # ---- pstart ----
    if pstart is None:
        pstart = pstart_heuristic(x, strata_split, calib)

    if verbose:
        nrs_display = len(strata_split)
        stratum_word = "stratum" if nrs_display == 1 else "strata"
        print(f"vsn2: {nr} x {nc} matrix ({nrs_display} {stratum_word}).")

    # ---- build VsnInput ----
    v = VsnInput(
        x=x,
        reference=ref,
        strata=strata_arr,
        ordered=False,
        lts_quantile=lts_quantile,
        subsample=subsample,
        verbose=verbose,
        calib=calib,
        pstart=pstart,
        optimpar=opar,
    )

    # ---- fit ----
    res = vsn_sample(v)

    # ---- hoffset ----
    cof = res.coefficients if ref is None else reference.coefficients  # type: ignore[union-attr]
    # cof[:,:,1] are the log-scale parameters; take row means across columns
    b_mean = np.nanmean(cof[:, :, 1], axis=1)  # shape (n_strata,)
    res.hoffset = np.log2(2 * scaling_factor_transformation(b_mean))

    # ---- transformed data ----
    if return_data:
        res.strata = strata_arr
        res.hx = vsn2_trsf(
            x=x,
            p=res.coefficients,
            strata=res.strata if len(res.strata) > 0 else np.ones(nr, dtype=int),
            hoffset=res.hoffset,
            calib=calib,
        )

    if verbose:
        print("Please use a mean-SD plot to verify the fit.")

    return res


def justvsn(
    x: np.ndarray,
    reference: Optional[VsnResult] = None,
    strata: Optional[np.ndarray] = None,
    lts_quantile: float = 0.9,
    subsample: int = 0,
    verbose: bool = False,
    calib: str = "affine",
    pstart: Optional[np.ndarray] = None,
    min_data_points_per_stratum: int = 42,
    optimpar: Optional[dict] = None,
    defaultpar: Optional[dict] = None,
) -> np.ndarray:
    """Fit VSN and return only the transformed matrix.

    This is the Python convenience equivalent of R's ``vsn::justvsn()``.
    It calls :func:`vsn_matrix` with ``return_data=True`` and returns the
    resulting ``hx`` matrix instead of the full :class:`VsnResult` object.

    Parameters
    ----------
    x : array-like
        Numeric matrix with features in rows and samples in columns. Missing
        observations must be represented by ``np.nan``.
    reference : VsnResult or None
        Optional fitted reference transformation.
    strata : array-like or None
        Optional one-based integer stratum labels, one per row.
    lts_quantile : float, default 0.9
        Fraction retained by least-trimmed-squares fitting.
    subsample : int, default 0
        Rows sampled per stratum. Zero uses all rows.
    verbose : bool, default False
        Print fitting diagnostics.
    calib : {"affine", "none"}, default "affine"
        Calibration model.
    pstart : array-like or None
        Optional starting coefficients.
    min_data_points_per_stratum : int, default 42
        Minimum rows required per stratum.
    optimpar : dict or None
        Overrides selected optimizer parameters.
    defaultpar : dict or None
        Overrides the default optimizer dictionary.

    Returns
    -------
    numpy.ndarray
        Float64 VSN-transformed matrix with the same shape and missing-value
        positions as ``x``.

    Notes
    -----
    This function does not treat zero as missing automatically. Convert zeros
    to ``np.nan`` before calling it when zero means an undetected intensity.
    Use :func:`vsn_matrix` when fitted coefficients or optimizer diagnostics
    are also required.
    """
    result = vsn_matrix(
        x=x,
        reference=reference,
        strata=strata,
        lts_quantile=lts_quantile,
        subsample=subsample,
        verbose=verbose,
        return_data=True,
        calib=calib,
        pstart=pstart,
        min_data_points_per_stratum=min_data_points_per_stratum,
        optimpar=optimpar,
        defaultpar=defaultpar,
    )

    if result.hx is None:
        raise RuntimeError("VSN fitting completed without transformed data.")

    return np.asarray(result.hx, dtype=np.float64)
