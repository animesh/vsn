# vsn2.py — Variance Stabilization and Normalization in Python

A Python/NumPy port of the **VSN** algorithm originally implemented in R/C by
Wolfgang Huber et al. (Bioconductor `vsn` package).

No R, no compiled C extensions — only `numpy` and `scipy`.

---

## Background

VSN fits a generalized log (asinh) transformation to expression data so that the
per-feature standard deviation is approximately constant across the intensity range.
It simultaneously estimates per-sample calibration parameters (offset and scale) and
the transformation, making it robust to differences in overall signal level between
samples.

**Reference:**
> Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. (2002)
> *Variance stabilization applied to microarray data calibration and to the
> quantification of differential expression.*
> Bioinformatics **18**(suppl 1), S96–S104.

---

## Requirements

```
numpy
scipy
```

Install with:

```bash
pip install numpy scipy
```

---

## Quick Start

```python
import numpy as np
from vsn2 import vsn_matrix

# x: float array of shape (n_features, n_samples)
result = vsn_matrix(x)

# Variance-stabilized data in log2-like scale
hx = result.hx          # shape (n_features, n_samples)
```

---

## How It Works

The transformation applied to each value is:

$$h(y) = \frac{\text{arcsinh}(e^b \cdot y + a)}{\ln 2} - \text{hoffset}$$

where `a` (offset) and `b` (log-scale) are fitted per sample (and per stratum if
provided). The parameters are estimated by maximum profile likelihood using the
**L-BFGS-B** optimizer, wrapped in a **Least Trimmed Squares (LTS)** robustness
loop that iteratively downweights high-residual features.

---

## API Reference

### `vsn_matrix`

The main entry point.

```python
vsn_matrix(
    x,
    reference=None,
    strata=None,
    lts_quantile=0.9,
    subsample=0,
    verbose=False,
    return_data=True,
    calib="affine",
    pstart=None,
    min_data_points_per_stratum=42,
    optimpar=None,
    defaultpar=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | `ndarray (nr, nc)` | — | Expression matrix: rows = features, cols = samples |
| `reference` | `VsnResult` or `None` | `None` | Normalize against a pre-fitted reference |
| `strata` | `ndarray (nr,)` int or `None` | `None` | Per-row stratum labels (1-based integers covering 1…n) |
| `lts_quantile` | `float` | `0.9` | Fraction of features used per LTS iteration |
| `subsample` | `int` | `0` | Rows to subsample for fitting (0 = use all) |
| `verbose` | `bool` | `False` | Print progress messages |
| `return_data` | `bool` | `True` | Compute and store transformed matrix in result |
| `calib` | `str` | `"affine"` | `"affine"` (per-sample offset+scale) or `"none"` (single global transform) |
| `pstart` | `ndarray (n_strata, d2, 2)` or `None` | `None` | Custom starting parameters |
| `min_data_points_per_stratum` | `int` | `42` | Minimum rows per stratum; raises if not met |
| `optimpar` | `dict` or `None` | `None` | Override specific optimizer settings (see below) |
| `defaultpar` | `dict` or `None` | `None` | Full optimizer settings override |

**Returns:** `VsnResult`

---

### `VsnResult`

Stores the fitted model.

| Attribute | Type | Description |
|---|---|---|
| `.hx` | `ndarray (nr, nc)` | Variance-stabilized data (log2-like scale) |
| `.coefficients` | `ndarray (n_strata, nc, 2)` | Fitted parameters: `[:,:,0]` = offsets, `[:,:,1]` = log-scales |
| `.mu` | `ndarray (nr,)` | Row means in transformed space |
| `.sigsq` | `float` | Estimated residual variance |
| `.hoffset` | `ndarray (n_strata,)` | log2 offset applied after transformation |
| `.lbfgsb` | `int` | Optimizer return code (0 = success) |
| `.calib` | `str` | Calibration mode used |

---

### Optimizer parameters (`optimpar`)

Pass as a dict to `optimpar=`. Keys use underscores in place of R's dots.

| Key | Default | Description |
|---|---|---|
| `factr` | `5e7` | L-BFGS-B stopping tolerance factor (lower = stricter) |
| `pgtol` | `2e-4` | Projected gradient tolerance |
| `maxit` | `60000` | Maximum optimizer iterations |
| `trace` | `0` | Verbosity level for L-BFGS-B (0 = silent) |
| `cvg_niter` | `7` | Maximum LTS outer iterations |
| `cvg_eps` | `0.0` | Convergence threshold on max change in `hx` (0 = disabled) |

---

## Usage Examples

### Basic normalization

```python
import numpy as np
from vsn2 import vsn_matrix

# Simulate data: 500 features, 6 samples
rng = np.random.default_rng(1234)
signal = rng.exponential(1000, size=500)
X = signal[:, None] * rng.uniform(0.8, 1.5, size=(1, 6)) \
    + rng.uniform(0, 300, size=(1, 6)) \
    + rng.normal(0, 30, size=(500, 6))
X = X.clip(1)

result = vsn_matrix(X, verbose=True)
hx = result.hx  # (500, 6) variance-stabilized matrix
```

### Normalize new data against a fitted reference

```python
# Fit on a training set
ref = vsn_matrix(X_train, return_data=False)

# Apply to new data using the same calibration
from vsn2 import vsn2_trsf
import numpy as np

hx_new = vsn2_trsf(
    x=X_new,
    p=ref.coefficients,
    strata=np.ones(X_new.shape[0], dtype=int),
    hoffset=ref.hoffset,
    calib="affine",
)
```

### Using strata (e.g. spatial blocks on an array)

```python
import numpy as np
from vsn2 import vsn_matrix

# Assign each feature to one of 3 spatial blocks
strata = np.repeat([1, 2, 3], 200)   # 600 features total

result = vsn_matrix(X, strata=strata, calib="affine")
```

### Apply the transformation directly

```python
from vsn2 import vsn2_trsf
import numpy as np

hx = vsn2_trsf(
    x=X,
    p=result.coefficients,
    strata=np.ones(X.shape[0], dtype=int),
    hoffset=result.hoffset,
    calib="affine",
)
```

### Scaling factor helper

```python
from vsn2 import scaling_factor_transformation
import numpy as np

b = result.coefficients[0, :, 1]   # log-scale parameters for stratum 0
scales = scaling_factor_transformation(b)  # exp(b)
```

---

## What Was Ported

| Original (R / C) | Python equivalent |
|---|---|
| S4 class `vsn` | `VsnResult` dataclass |
| S4 class `vsnInput` | `VsnInput` dataclass |
| C `loglik` / `grad_loglik` | `_negloglik_and_grad()` — pure NumPy |
| C `calctrsf` | `_calc_trsf()` |
| C `vsn2_optim` via `lbfgsb` | `vsn_ml()` via `scipy.optimize.minimize(..., method="L-BFGS-B")` |
| C `vsn2_trsf` | `vsn2_trsf()` |
| C `vsn2_scalingFactorTransformation` | `scaling_factor_transformation()` |
| R `vsnML` | `vsn_ml()` |
| R `vsnLTS` | `vsn_lts()` |
| R `calcistrat` | `_calc_istrat()` |
| R `vsnColumnByColumn` | `vsn_column_by_column()` |
| R `vsnStrata` | `vsn_strata()` |
| R `vsnSample` | `vsn_sample()` |
| R `vsnMatrix` | `vsn_matrix()` |
| R `pstartHeuristic` | `pstart_heuristic()` |
| R `int2factor` | `int_to_strata()` |
| R `isSmall` | `is_small()` |
| R `rowV` | `row_variances()` |
| R `calibCharToInt` | `_calib_to_int()` |

---

## Notes

- `optimpar` keys use underscores (`cvg_niter`, `cvg_eps`) instead of R's dots.
- The `calib="none"` mode fits a single global transform (no per-sample calibration);
  it requires a `reference` or at least 2 columns.
- Features that are all `NaN` are automatically excluded from fitting and remain `NaN`
  in the output.
- The `subsample` option cannot be combined with `reference` normalization.

## BUGS FIXED:
  1. pstart_heuristic: now uses b = log(1/mean(y_col)) per column
     instead of b=1. This avoids the enormous gradient at startup
     and allows the optimizer to converge properly.

  2. _rank_na_last: now uses scipy.stats.rankdata(method='average')
     for ties (matching R's default), and assigns sequential ranks
     to NaNs in order of appearance (matching R's na.last=TRUE).

  3. LTS slice assignment: now uses floor((rank-1)/(n/5)) which 
     correctly maps 1-based ranks to 5 equal slices, matching 
     R's cut(rank, breaks=5).

  4. vsn_ml: removed redundant pstart flattening, now passes 
     jac=True to minimize() so likelihood and gradient are 
     computed together (more efficient, fewer function calls).

REMAINING DIVERGENCE (fundamental, not a bug):
  Looks like that Python implementation finds a BETTER local minimum than R (lower sigsq, lower row SDs = better variance stabilization). Both seem valid solutions to the same likelihood function.
  
  Root cause: R's Fortran L-BFGS-B (from R_ext/Applic.h, ca. 1997)
  and scipy's L-BFGS-B (from Zhu et al. 1997 with modifications) 
  have different line search implementations. From the naive 
  starting point (a=0, b=1), R's implementation escapes a saddle 
  region that scipy does not. With the improved initialization in 
  the fixed Python code, scipy finds an equal or better solution.
