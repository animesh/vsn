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

## BUGS FIXED:


---

**Bug 1 - CRITICAL: `pstart_heuristic` sets wrong `log_b` init (line ~751)**

```python
# WRONG (original):
pstart[:, :, 1] = 1.0   # => b_init = exp(1) ≈ 2.718, not 1
# R's pstartHeuristic returns b=1, meaning log_b=0, not 1
```

This is an off-by-`e` error - the parameterization uses `b = exp(log_b)`, so setting `log_b=1` gives `b=e` not `b=1`.

---

**Bug 2 - CRITICAL for proteomics: initialization too far from optimum**

Even with `log_b=0` (b=1), for proteomics intensities of ~10^6-10^8, the argument `u = b*y ≈ 10^6`. The optimizer gets trapped in a pure-log-regime local optimum where variance stabilization FAILS. Demonstrated concretely: mean-SD correlation was **-0.74** (broken) vs **+0.02** (fixed) after switching to MAD-based initialization:

```python
b_j = 1 / (2 * median_j(y))   # puts median intensity at u ≈ 0.5 (arcsinh transition)
a_j = -0.5
```

The b values actually recovered the exact load ratios (1.3/0.7 = 1.857) confirming the optimizer found the correct solution.

---

**Bug 3 - MINOR: fragile `assert` in `vsn_lts` (line ~603)**

```python
assert is_small(rsv.mu - hmean), "mu mismatch"  # tolerance = sqrt(eps) ≈ 1.5e-8
```

This is an internal consistency check but could fire on near-singular data or float edge cases. Changed to `warnings.warn()`.

---

- NLL formula (profile likelihood is correct)
- Gradient formula - the 4.5e-3 relative error at eps=1e-6 looked suspicious but it's just catastrophic cancellation in the finite-difference itself; at eps=1e-4 the error is 1e-5, confirming the gradient is correct
- `istrat` construction, `_calc_trsf` indexing, `hoffset` formula, `vsn2_trsf` scaling

One assumption to flag: your simulation assumed noise ~ sqrt(y) (Poisson regime). If your real data has different noise structure, the mean-SD plot is the right diagnostic to run on actual data after applying this.

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

hosted at [posit-cloud](https://fuzzylife-vsn.share.connect.posit.cloud/), for local serve 
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
