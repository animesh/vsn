# vsn2.py — Variance Stabilization and Normalization in Python

A Python/NumPy port of the **VSN** algorithm originally implemented in R/C by Wolfgang Huber et al. (Bioconductor `vsn` package).

No R, no compiled C extensions — only `numpy` and `scipy`.

------------------------------------------------------------------------

## Background

VSN fits a generalized log (asinh) transformation to expression data so that the per-feature standard deviation is approximately constant across the intensity range. It simultaneously estimates per-sample calibration parameters (offset and scale) and the transformation, making it robust to differences in overall signal level between samples.

**Reference:** \> Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. (2002) \> *Variance stabilization applied to microarray data calibration and to the \> quantification of differential expression.* \> Bioinformatics **18**(suppl 1), S96–S104.

------------------------------------------------------------------------

## Requirements

```         
numpy
scipy
```

Install with:

``` bash
pip install numpy scipy
```

------------------------------------------------------------------------

## Quick Start

``` python
import numpy as np
from vsn2 import vsn_matrix

# x: float array of shape (n_features, n_samples)
result = vsn_matrix(x)

# Variance-stabilized data in log2-like scale
hx = result.hx          # shape (n_features, n_samples)
```

------------------------------------------------------------------------

## How It Works

The transformation applied to each value is:

$$h(y) = \frac{\text{arcsinh}(e^b \cdot y + a)}{\ln 2} - \text{hoffset}$$

where `a` (offset) and `b` (log-scale) are fitted per sample (and per stratum if provided). The parameters are estimated by maximum profile likelihood using the **L-BFGS-B** optimizer, wrapped in a **Least Trimmed Squares (LTS)** robustness loop that iteratively downweights high-residual features.

------------------------------------------------------------------------

## R C code vs Python, diff function by function

1. NLL formula (vsn2.c loglik vs Python _negloglik_and_grad)
R vsn2.c:
  jac1 += log(1.0+z*z);  jac2 += ni*log(bj);   jacobian = jac1*0.5 - jac2;
  sigsq = ssq/nt;  residuals = nt/2.0;
  ll = nt/2*log(2π*sigsq) + nt/2 + jacobian;

Python:
  jac1 += np.sum(np.log(1.0 + z**2));  jac2 += np.sum(valid)*np.log(bj)
  jacobian = 0.5*jac1 - jac2;  sigsq = ssq/nt;  residuals = nt/2.0
  ll = nt/2*log(2π*sigsq) + nt/2 + jacobian

2. Gradient (vsn2.c grad_loglik vs Python)
R vsn2.c:
  z = resid[k]/sigsq + ma[k]*ly[k]    # (r/σ² + m*u)
  sa += z * ma[k]                      # Σ (r/σ² + m*u)*m = Σ r*m/σ² + m²*u
  sb += z * ma[k] * y[k]              # Σ (r/σ² + m*u)*m*y
  gr[j]       = sa
  gr[j+nrs]   = exp(b[j]) * (sb - nj/exp(b[j]))

Python:
  z   = rv*rfac + mv*lyv               # same
  sa  = sum(z * mv)                    # same
  sb  = sum(z * mv * yv)               # same
  grad[j]    = sa
  grad[j+ns] = exp(b[j]) * (sb - nj/bj)

3. pstart heuristic
R pstartHeuristic:   pstart[,,2] = 1   → log_b = 1 → b = exp(1) ≈ 2.718
Python pstart_heuristic: pstart[:,:,1] = 1.0 → same

4. hoffset
R:  hoffset = log2(2 * scalingFactorTransformation(rowMeans(cof[,,2])))
    scalingFactorTransformation(b) = exp(b)   [vsn2.c FUN macro]
    → log2(2 * exp(mean(log_b)))

Python: log2(2 * exp(nanmean(cof[:,:,1], axis=1)))
(rowMeans=nanmean only differs if coefficients have NaN)

5. L-BFGS-B settings
R vsn2.c:   lmm=5, factr=5e7, pgtol=2e-4, maxit=60000, bounds:a=unbounded, b=[-100,100]
Python:  maxcor=5, factr*eps for ftol, gtol=pgtol=2e-4, maxiter=60000, bounds: same

6. LTS slicing - ACTUAL DIFFERENCE FOUND
R vsnLTS:
  facslice = cut(rank(hmean, na.last=TRUE), breaks=5)
  grquantile = tapply(rvar, list(facslice, facstrata), quantile, probs=0.9, na.rm=TRUE)
  whsel = which( (rvar <= grquantile[cbind(slice,intstrata)]) | (slice==1) )

  'cut(x, breaks=5)' uses RIGHT-CLOSED intervals: (a,b]
  The LOWEST element is included in the first interval (special case).
  
  Boundary values (exact bin edges) go to the UPPER bin in R.

Python _rank_na_last + np.searchsorted:
  np.searchsorted(cut_breaks, rank_hmean, side='left') - 1
  This uses LEFT-CLOSED intervals: [a,b)
  Boundary values go to the LOWER bin in Python.

BOUNDARY HANDLING DIFFERS
  Values exactly on a bin boundary are assigned to DIFFERENT slices.
  In practice: affects ~0-5 proteins per LTS iteration.
  Cumulative over 7 iterations: slightly different whsel → different fitted params.

7. quantile method
R:   quantile(x, probs=0.9, na.rm=TRUE, type=7)  [R default type]
NumPy: np.nanquantile(x, 0.9)  [uses linear interp = R type 7]

8. rowMeans NA handling during rvar
R:   rvar = rowSums((hy - hmean)^2)   [na.rm=FALSE by default → NaN propagates]
Py:  rvar = np.sum((hy - hmean[:,None])**2, axis=1) [NaN propagates too]

SUMMARY
ONE substantive algorithmic difference:
  R cut() RIGHT-CLOSED intervals vs Python LEFT-CLOSED (searchsorted)
  → slightly different slice assignments for values exactly on bin boundaries
  → different LTS selection over 7 iterations
  → slightly different final (a,b) parameters → RMSE ≈ 0.0004

NLL, gradient, pstart, hoffset, L-BFGS-B settings same

## API Reference

### `vsn_matrix`

The main entry point.

``` python
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
|------------------|------------------|------------------|------------------|
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

------------------------------------------------------------------------

### `VsnResult`

Stores the fitted model.

| Attribute | Type | Description |
|------------------------|------------------------|------------------------|
| `.hx` | `ndarray (nr, nc)` | Variance-stabilized data (log2-like scale) |
| `.coefficients` | `ndarray (n_strata, nc, 2)` | Fitted parameters: `[:,:,0]` = offsets, `[:,:,1]` = log-scales |
| `.mu` | `ndarray (nr,)` | Row means in transformed space |
| `.sigsq` | `float` | Estimated residual variance |
| `.hoffset` | `ndarray (n_strata,)` | log2 offset applied after transformation |
| `.lbfgsb` | `int` | Optimizer return code (0 = success) |
| `.calib` | `str` | Calibration mode used |

------------------------------------------------------------------------

### Optimizer parameters (`optimpar`)

Pass as a dict to `optimpar=`. Keys use underscores in place of R's dots.

| Key | Default | Description |
|------------------------|------------------------|------------------------|
| `factr` | `5e7` | L-BFGS-B stopping tolerance factor (lower = stricter) |
| `pgtol` | `2e-4` | Projected gradient tolerance |
| `maxit` | `60000` | Maximum optimizer iterations |
| `trace` | `0` | Verbosity level for L-BFGS-B (0 = silent) |
| `cvg_niter` | `7` | Maximum LTS outer iterations |
| `cvg_eps` | `0.0` | Convergence threshold on max change in `hx` (0 = disabled) |

------------------------------------------------------------------------

## Usage Examples

### Basic normalization

``` python
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

``` python
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

``` python
import numpy as np
from vsn2 import vsn_matrix

# Assign each feature to one of 3 spatial blocks
strata = np.repeat([1, 2, 3], 200)   # 600 features total

result = vsn_matrix(X, strata=strata, calib="affine")
```

### Apply the transformation directly

``` python
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

``` python
from vsn2 import scaling_factor_transformation
import numpy as np

b = result.coefficients[0, :, 1]   # log-scale parameters for stratum 0
scales = scaling_factor_transformation(b)  # exp(b)
```

------------------------------------------------------------------------

## What Was Ported

| Original (R / C) | Python equivalent |
|------------------------------------|------------------------------------|
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

------------------------------------------------------------------------

## Notes

- `optimpar` keys use underscores (`cvg_niter`, `cvg_eps`) instead of R's dots.
- The `calib="none"` mode fits a single global transform (no per-sample calibration); it requires a `reference` or at least 2 columns.
- Features that are all `NaN` are automatically excluded from fitting and remain `NaN` in the output.
- The `subsample` option cannot be combined with `reference` normalization.

## BUGS in vsn2mo.py vs correct vsn2.py + R notebook

BUG 1 [CRITICAL - wrong scale]: predict() missing /log(2)
  Line 78: return res - self.hoffset
  Should:  return res / np.log(2) - self.hoffset
  Effect:  output in raw arcsinh units (~32-35) not log2 scale (~6-22)
  Visible: Image 2 x-axis 32-35 instead of expected 17-22

BUG 2 [CRITICAL - wrong math]: NLL formula is wrong
  Line 149: nll = 0.5 * nc * np.nansum(np.log(ss)) - np.nansum(log_jac)
  This is: sum_over_rows(log(per_row_SS)) * nc/2  -- sum-of-logs
  vsn2.c:  nt/2 * log(total_SSQ/nt) + nt/2 + jacobian  -- log-of-sum
  These are mathematically different objectives.

BUG 3 [CRITICAL - wrong math]: Gradient uses per-row weights
  Line 165: dnll_dy = nc * res[:, j] / ss   (ss is per-row vector)
  vsn2.c:   rfac = 1.0 / sigsq              (sigsq is scalar total variance)
  Effect:   optimizer walks a different loss surface, wrong parameters.

BUG 4 [CRITICAL - missing]: No LTS robustification
  The marimo vsn_sample calls minimize() exactly ONCE.
  R's vsnLTS runs 7 iterations, trimming bottom-10%% rows each time.
  Effect:   non-robust fit, outlier proteins distort normalization.

BUG 5 [significant]: pstart_heuristic uses 1/median instead of exp(1)
  Line 109: b_init = 1.0 / (np.nanmedian(col_valid) + 1e-8)
  Correct:  log_b = 1.0  →  b = exp(1) ≈ 2.718  (matches R pstartHeuristic)
  Effect:   optimizer starts in wrong basin → RMSE=0.07 vs R (proven earlier).

BUG 6 [significant]: hoffset uses arithmetic mean of b, not geometric mean
  Line 280-281: b_mean = np.nanmean(b_vals); hoffset = log2(2*b_mean)
  R vsn2.c:     hoffset = log2(2 * exp(mean(log_b)))  = log2(2*geomean(b))
  These differ when b values vary across columns.

BUG 7 [wrong]: log_jac formula mixes parameterizations
  Line 143: log_jac = log(|b|) - 0.5*log(1+u^2)
  The vsn_obj_func bounds enforce b>1e-10 (b is scale, not log_b)
  But hoffset/predict treat b as already exponential.
  vsn2.c uses log_b as parameter: jacobian = 0.5*Σlog(1+u²) - Σnj*log_b_j

BUG 8 [visual]: Diagnostics mean-SD plot uses raw mean on x-axis
  R's meanSdPlot() default (ranks=TRUE) uses rank(rowMean) on x-axis.
  Marimo uses actual mean value. Minor but doesn't match R output.

BUGS CAUSING THE VISIBLE PROBLEMS:
  Image 2 x-axis 32-35 → Bug 1 (missing /log(2)) + Bug 5 (wrong init)
  Spike in running median at high end → Bugs 2,3,4 (wrong loss surface, no LTS)

## BUGS in vsn2mo.py fixed 

Bug 1 [CRITICAL - WRONG NLL]:
  vsn2mo.py line 149:
    nll = 0.5 * nc * np.nansum(np.log(ss))  - np.nansum(log_jac)
    # ss = per-ROW sum-of-squares; logs them individually then sums
  Correct (vsn2.c loglik):
    nll = (nt/2)*log(2π*SSQ/nt) + nt/2 + jacobian
    # SSQ = TOTAL sum of squares across all rows; single log

Bug 2 [CRITICAL - WRONG GRADIENT]:
  vsn2mo.py line 165:
    dnll_dy = nc * res[:, j] / ss    # ss is per-row, shape (nr,)
  Correct (vsn2.c grad_loglik):
    rfac = 1/sigsq   # sigsq = SSQ/nt (scalar)
    z = r*rfac + ma*ly   # (r/σ² + m*u)
    grad_a = sum(z*ma); grad_logb = exp(b)*(sum(z*ma*y) - nj/exp(b))

Bug 3 [CRITICAL - WRONG LOG_B PARAMETERIZATION]:
  vsn2mo.py lines 143, 163:
    log_jac[:, j] = np.log(np.abs(b) + 1e-12) - 0.5*np.log(1+val**2)
    dlog_db = 1.0/(b+1e-12) - x[:,j]*val/(1+val**2)
    # b is bounded > 1e-10 directly, optimizer treats it as raw scale
  Correct (vsn2.c FUN macro):
    b_param = log_b  (unconstrained, bounded -100..100)
    b = exp(b_param)
    jac2 = n_j * b_param  # n_j * log(b) = n_j * log_b
    grad_logb = exp(b_param) * (sb - nj/exp(b_param))

Bug 4 [CRITICAL - WRONG PSTART]:
  vsn2mo.py line 109:
    b_init = 1.0 / (np.nanmedian(col_valid) + 1e-8)  # MAD-based, WRONG
  Correct (R pstartHeuristic):
    pstart[:,:,1] = 1.0   # log_b = 1 → b = exp(1) ≈ 2.718

Bug 5 [CRITICAL - WRONG HOFFSET]:
  vsn2mo.py lines 280-281:
    b_mean = np.nanmean(b_vals)           # arithmetic mean of b
    res.hoffset = np.log2(2.0 * b_mean)  # log2(2*mean(b)) WRONG
  Correct (R vsnMatrix):
    hoffset = log2(2 * scalingFactorTransformation(mean(log_b)))
            = log2(2 * exp(mean(log_b)))  # geometric mean of b

Bug 6 [CRITICAL - MISSING /log(2) IN PREDICT]:
  vsn2mo.py line 78:
    return res - self.hoffset     # res = arcsinh(a + b*x), no /log(2)!
  Correct (R vsn2trsf with hoffset):
    hx = arcsinh(a + exp(log_b)*x)
    hx = hx / log(2) - hoffset     # divide by log(2) → log2 scale

Bug 7 [NO LTS ITERATIONS]:
  vsn2mo.py: single L-BFGS-B call, no robust LTS loop
  Correct (R vsnLTS): 7 iterations, each time trimming 10% worst-residual rows

Bug 8 [WRONG MEAN-SD PLOT X-AXIS]:
  vsn2mo.py line 383:
    _row_means = np.nanmean(normalized_mat, axis=1)  # actual values on x-axis
  Correct R meanSdPlot(ranks=TRUE):
    x-axis = rank(rowMeans) / n   # ranks normalized to [0,1]

Net effect of Bugs 5+6: output scale is arcsinh(b*x) - log2(2*mean(b))
instead of arcsinh(b*x)/log(2) - log2(2*exp(mean(log_b)))
This shifts values by ~22 log2 units → explains 32-35 range in plot

## Issues still with manifest.json generated with https://docs.posit.co/connect-cloud/how-to/r/dependencies.html

Auto deployment via positron failed again because the system could not download the **`BiocGenerics`** package from Bioconductor. The specific HTTP **404 (Not Found)** error was returned when attempting to fetch:

```
https://bioconductor.org/packages/3.21/books/src/contrib/BiocGenerics_0.54.0.tar.gz
```

### Root Cause
There are **two related problems**:

1. **Wrong Bioconductor repository path** — The URL contains `/books/` in the path, but `BiocGenerics` is a **software** package, not a book. The correct URL should use `/bioc/` instead:
   ```
   # Incorrect (404)
   .../packages/3.21/books/src/contrib/BiocGenerics_0.54.0.tar.gz

   # Correct
   .../packages/3.21/bioc/src/contrib/BiocGenerics_0.54.0.tar.gz
   ```
check https://bioconductor.org/packages/release/bioc/html/BiocGenerics.html

2. **Content type mismatch** — The logs also warn that the project was published as `quarto-static` but the `manifest.json` declares it as `quarto-shiny`. This suggests the manifest may be outdated or misconfigured.

---

## Suggested Fixes

### 1. Fix the `manifest.json` or `renv.lock`
Check how `BiocGenerics` is referenced in your `manifest.json` or `renv.lock` and correct the repository URL:

```json
// In renv.lock, ensure the source is correct:
"BiocGenerics": {
  "Package": "BiocGenerics",
  "Version": "0.54.0",
  "Source": "Bioconductor",
  "Repository": "https://bioconductor.org/packages/3.21/bioc"
}
```

### 2. Regenerate the Manifest Locally
Regenerate your deployment manifest to ensure correct package sources are captured:

```r
# In R, from your project directory:
renv::snapshot()
# or
rsconnect::writeManifest()
```

### 3. Fix the Content Type Mismatch
Align your `manifest.json` content type with what you're actually deploying. If it is a Shiny app, set:

```json
{
  "metadata": {
    "appmode": "quarto-shiny"
  }
}
```

### 4. Verify Bioconductor Version Compatibility
Ensure Bioconductor 3.21 is compatible with your R version:

```r
BiocManager::version()        # Check current version
BiocManager::valid()          # Check for inconsistencies
```


## run.py: generic VSN2 runner

`python run.py input.tsv output.tsv --id-columns "Protein.Group,Protein.Names,Genes" --intensity-columns "Sample1,Sample2,Sample3,Sample4"`

For the DIA-NN matrix: `python run.py "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsv" --id-columns "Protein.Group,Protein.Names,Genes,First.Protein.Description,N.Sequences,N.Proteotypic.Sequences" --intensity-regex "^F:" report.pg.vsn2.tsv`

Explicit names: --id-columns "ID1,ID2" --intensity-columns "Sample1,Sample2,Sample3"

Regular expressions: --id-regex "\^Protein" --intensity-regex "\^Intensity\_" --intensity-regex "\^Control\_" --intensity-regex "\^Treatment\_"

Supports CSV and TSV input. Preserves specified ID columns. Returns only ID columns followed by VSN-transformed intensity columns. Converts blank, infinite and nonnumeric intensity entries to NA. Treats zero as missing by default, matching your R workflow. Use --keep-zero to transform zeros instead. Preserves nonmissing values in partially missing rows. Writes fitted parameters to

<output>

.vsn2_parameters.csv. Requires at least two intensity columns.

## compare.py: generic VSN2 run.py comparison script

`python compare.py "report.pg.vsn2.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv"`

Optional row-level differences: --details-output "vsn_comparison_by_sample.csv" \^ --differences-output "vsn_differences_by_row.csv"

It reports: Matched intensity columns Compared finite values Overall RMSE Overall MAE Overall mean difference Maximum absolute error Per-sample Pearson correlation Per-sample R-squared Missing-value counts

## run_and_compare.py: generic VSN2 runner and comparison script

```         
python run_and_compare.py "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv" --python-output "python_vsn_matched_samples.tsv"
vsn2: 10991 x 20 matrix (1 stratum).
L:\promec\Animesh\Download\vsn\vsn2.py:936: UserWarning: 1584 rows were removed since they contained only NA elements.
  warnings.warn(f"{num_na} rows were removed since they contained only NA elements.")
Please use a mean-SD plot to verify the fit.

Generated Python VSN2 values
rows: 10991
id_columns: 6
intensity_columns: 20
rows_all_missing_in_selected_intensities: 1584
finite_input_values: 133623
finite_output_values: 133623
optimizer_return_code: 0
output: python_vsn_matched_samples.tsv
parameters: python_vsn_matched_samples.vsn2_parameters.csv

Comparison with R output
Rows in each file: 10991
Matched intensity columns: 20
Compared finite values: 133623
Overall RMSE: 0.0702498174389
Overall MAE: 0.0425473873035
Overall mean difference (Python - R): 0.0322643754797
Overall maximum absolute error: 1.72702674977
Unmatched Python numeric columns: 2
Unmatched R intensity columns: 0
Per-sample details: vsn_matched_comparison_by_sample.csv

The saved Python output can be compared again later with:
python compare.py "python_vsn_matched_samples.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv"
```

### check

```         
python compare.py "python_vsn_matched_samples.tsv" "L:\promec\TIMSTOF\LARS\2026\260518_Sonali\DIANNv2P2.63.260612_140833.64.highacc\report.pg_matrix.tsvLFQvsn0.250.5Rem20Groups.txtLFQvsnF..promec.TIMSTOF.LARS.2026.260518_Sonali.260518_Sonali_CorTestBH.csv"
Rows in each file: 10991
Matched intensity columns: 20
Compared finite values: 133623
Overall RMSE: 0.000431713355128
Overall MAE: 0.000262345883364
Overall mean difference (Python - R): 1.55570290982e-05
Overall maximum absolute error: 0.00155267994017
Unmatched Python numeric columns: 2
Unmatched R intensity columns: 0
Per-sample details: vsn_output_comparison_by_sample.csv
```

Looks like the ~0.000432 RMSE is irreducible without matching R's cut() boundary convention but quality of variance stabilization (mean-SD corr 0.019 = R's 0.019)

## run.r: generic VSN2 R script

```         
Rscript run.r proteinGroups.txt "LFQ"

[1] "USAGE:<path to>Rscript diffExprTestCor.r <complete path to proteinGroups.txt> and <intensity columns to consider>"
[1] "supplied argument(s): 2"
[1] ".\\proteinGroups.txt" "LFQ"
  [1] "Protein.IDs"
  [2] "Majority.protein.IDs"
  [3] "Peptide.counts..all."
  [4] "Peptide.counts..razor.unique."
  [5] "Peptide.counts..unique."
  [6] "Protein.names"
  [7] "Gene.names"
  [8] "Fasta.headers"
  [9] "Number.of.proteins"
 [10] "Peptides"
 [11] "Razor...unique.peptides"
 [12] "Unique.peptides"
 [13] "Library.indices"
 [14] "Majority.library.index"
 ...
 LFQ.intensity.l50_100spd_OT_1ulirt_S2.C2_1_6365
 Min.   :  3855
 1st Qu.: 31665
 Median : 52601
 Mean   : 74927
 3rd Qu.:138800
 Max.   :156300
 NA's   :1
Warning message:
Removed 2 rows containing missing values or values outside the scale range (`geom_line()`).
Warning message:
Removed 2 rows containing missing values or values outside the scale range (`geom_line()`).
```         


## run.qmd: generic quarto script using shiny server

hosted at [posit-cloud](https://fuzzylife-vsn.share.connect.posit.cloud/), for serving locally try
```         
quarto serve run.qmd                                                                                                                      

Loading required namespace: shiny


processing file: run.qmd
1/24
2/24 [unnamed-chunk-1]
3/24 [unnamed-chunk-2] 
4/24
5/24 [unnamed-chunk-3]
6/24
7/24 [unnamed-chunk-4]
8/24
9/24 [unnamed-chunk-5]
10/24
11/24 [unnamed-chunk-6]
12/24
13/24 [unnamed-chunk-7]
14/24
15/24 [unnamed-chunk-8]
16/24
17/24 [unnamed-chunk-9]
18/24
19/24 [unnamed-chunk-10]
20/24
21/24 [unnamed-chunk-11]
22/24 [unnamed-chunk-12]
23/24 [unnamed-chunk-13]
24/24
output file: run.knit.md

pandoc 
  to: html
  output-file: run.html
  standalone: true
  section-divs: true
  html-math-method: mathjax
  wrap: none
  default-image-extension: png
  toc: true
  variables: {}

metadata
  document-css: false
  link-citations: true
  date-format: long
  lang: en
  engines:
    - path: c:\Users\animeshs\Positron\resources\app\quarto\share\extension-subtrees\julia-engine\_extensions\julia-engine\julia-engine.js
  title: Variance Stabilizing Normalization (VSN)
  server:
    type: shiny
  theme: cosmo

Loading required package: shiny
Warning: package 'shiny' was built under R version 4.5.1
Loading required package: Biobase
Loading required package: BiocGenerics
Loading required package: generics

Attaching package: 'generics'

The following objects are masked from 'package:base':

    as.difftime, as.factor, as.ordered, intersect, is.element, setdiff,
    setequal, union


Attaching package: 'BiocGenerics'

The following objects are masked from 'package:stats':

    IQR, mad, sd, var, xtabs

The following objects are masked from 'package:base':

    anyDuplicated, aperm, append, as.data.frame, basename, cbind,
    colnames, dirname, do.call, duplicated, eval, evalq, Filter, Find,
    get, grep, grepl, is.unsorted, lapply, Map, mapply, match, mget,
    order, paste, pmax, pmax.int, pmin, pmin.int, Position, rank,
    rbind, Reduce, rownames, sapply, saveRDS, table, tapply, unique,
    unsplit, which.max, which.min

Welcome to Bioconductor

    Vignettes contain introductory material; view with
    'browseVignettes()'. To cite Bioconductor, see
    'citation("Biobase")', and for packages 'citation("pkgname")'.

Warning: package 'DT' was built under R version 4.5.1

Attaching package: 'DT'

The following objects are masked from 'package:shiny':

    dataTableOutput, renderDataTable

Browse at http://localhost:6486/
```         

## vsn2mo.py: marimo notebook script using shiny server

hosted at [molab-wasm-page](https://molab.marimo.io/github/animesh/vsn/blob/master/vsn2mo.py/wasm), for serving locally try
```
uv run marimo run vsn2mo.py 
This notebook has inlined package dependencies.
Run in a sandboxed venv containing this notebook's dependencies? [Y/n]: 
Running in a sandbox: /home/animeshs/.local/bin/uv run --isolated --no-project --compile-bytecode --with-requirements /tmp/tmpq0ilqsa0.txt --python >=3.13 marimo run vsn2mo.py

        Running vsn2mo.py ⚡

        ➜  URL: http://localhost:2718

        💡 Tip: Pair-program with AI agents on running notebooks
                Guide: https://links.marimo.app/marimo-pair
```         
