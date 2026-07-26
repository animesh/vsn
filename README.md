# VSN in Python

A NumPy/SciPy implementation of variance stabilization and normalization (VSN), based on the [Bioconductor `vsn` package](https://github.com/Huber-group-EMBL/vsn) by Wolfgang Huber and contributors.

The implementation fits an additive-multiplicative error model, estimates calibration parameters, applies the generalized logarithm, and uses the same robust least-trimmed-squares workflow as the R implementation.

The implementation reproduces native R `vsn::justvsn()` results to floating-point precision across dense, sparse, noisy, and high-missingness validation scenarios. In a deliberate strong-outlier simulation, the largest observed absolute Python versus R difference was approximately `1.11e-7`.

## Installation

Install the [published](https://pypi.org/project/vsn/) package from PyPI:

```bash
python -m pip install vsn
```

or current repository version directly from GitHub:

```bash
python -m pip install "git+https://github.com/animesh/vsn.git"
```

Verify the installation:

```bash
python -c "import vsn; print(vsn.__file__)"
python -c "from vsn import justvsn; print('justvsn available')"
```

## Quick start with `justvsn()`

The simplest interface mirrors R's `vsn::justvsn()` and returns the transformed matrix directly:

```python
import numpy as np
from vsn import justvsn

x = np.array(
    [
        [1200.0, 1500.0, np.nan],
        [2500.0, 2300.0, 2700.0],
        [5000.0, np.nan, 4800.0],
        [9000.0, 8700.0, 9200.0],
        [16000.0, 15500.0, 17000.0],
    ],
    dtype=np.float64,
)

normalized = justvsn(
    x,
    min_data_points_per_stratum=0,
)
```

The lower minimum is used only because this example has five rows. For ordinary datasets with at least 42 rows per stratum, use:

```python
normalized = justvsn(x)
```

`justvsn()` returns a `float64` NumPy array with the same shape and missing-value positions as the input.

## Development installation

Clone the repository and install an editable checkout:

```bash
git clone https://github.com/animesh/vsn.git
cd vsn
python -m pip install -e .
```

Editable installation links the environment to the local `vsn.py`. Local code changes become available after restarting the Python process.

Remove the installation with:

```bash
python -m pip uninstall vsn
```

## Requirements

The installed `vsn` module requires:

```text
Python >= 3.9
NumPy >= 1.23
SciPy >= 1.9
```

These dependencies are installed automatically by pip.

The matrix runner and comparison utilities also require pandas:

```bash
python -m pip install pandas
```

The standalone Marimo dashboard declares its own dependencies inline and can be run with `uv`.

## Project components

| File | Purpose |
|---|---|
| `vsn.py` | Reusable Python VSN implementation, including `justvsn()`, `vsn_matrix()`, and transformation utilities |
| `vsn2mo.py` | Standalone interactive Marimo normalization dashboard |
| `run.py` | Run VSN on selected columns of a CSV or TSV matrix |
| `compare.py` | Compare saved Python VSN output with R output |
| `run_and_compare.py` | Select matching samples, run Python VSN, save output, and compare with R |
| `simulate_and_run_python.py` | Generate reproducible simulation matrices and normalize every scenario with Python VSN |
| `run_simulations_r.R` | Normalize the same simulation matrices with native R `vsn::justvsn()` |
| `compare_simulation_results.py` | Compare R and Python simulation outputs and generate scenario-level and sample-level validation tables |
| `run_all_simulations.bat` | Run the complete simulation and comparison workflow on Windows |
| `run.qmd` | Quarto Shiny dashboard using the R `vsn` package |
| `pyproject.toml` | Python package metadata, build configuration, and dependencies |

The comparison workflow writes:

- `vsn_comparison_summary.tsv`: overall comparison metrics
- `vsn_comparison_by_sample.tsv`: per-sample comparison metrics
- `vsn_differences.tsv`: row-level Python-minus-R differences

For example, over results from https://fuzzylife.substack.com/p/proteomics-data-processing-with-maxquant 
```
curl -o proteinGroups.txt "https://zenodo.org/records/14557756/files/proteinGroups.txt?download=1"
Rscript run.r "proteinGroups.txt" LFQ                      
python run.py --id-columns "Protein IDs" --intensity-regex "LFQ " proteinGroups.txt proteinGroups.vsn.txt
python compare.py
```
shows `maximum absolute error: 6.03961325396e-14`
```text
Row alignment: IDs: Python 'Protein IDs', R 'Protein.IDs'
Aligned rows: 7816
Matched normalized columns: 7
Compared finite values: 18023
Missing-value mismatches: 0
Overall RMSE: 2.95738310314e-14
Overall MAE: 2.54391685484e-14
Overall mean difference (Python - R): -9.0468731226e-16
Overall maximum absolute error: 6.03961325396e-14
Summary: vsn_comparison_summary.tsv
Per-sample details: vsn_comparison_by_sample.tsv
Row-level differences: vsn_differences.tsv
```

The simulation workflow `run_all_simulations.bat` writes generated inputs, normalized matrices, environment information, and comparison tables under `vsn_simulation_results/`.

Manual sequence:

```bash
python simulate_and_run_python.py
Rscript run_simulations_r.R vsn_simulation_results
python compare_simulation_results.py vsn_simulation_results
```

The workflow uses the same saved matrices for native R and Python and records package versions and optimizer return codes. The principal validation outputs are:

- `comparison_summary.tsv`
- `comparison_by_sample.tsv`
- `comparison_ranked_by_max_error.tsv`
- `python_run_summary.tsv`
- `r_run_summary.tsv`
- `python_environment.json`
- `r_session_info.txt`

## Full fit with `vsn_matrix()`

Use `vsn_matrix()` when fitted coefficients, offsets, optimizer status, or other model details are required:

```python
import numpy as np
from vsn import vsn_matrix

result = vsn_matrix(x)

normalized = result.hx
coefficients = result.coefficients
optimizer_code = result.lbfgsb
```

An optimizer code of zero indicates successful convergence.

## Input orientation and data types

The input must be a two-dimensional numeric matrix with:

- rows representing features, proteins, or probes
- columns representing samples
- missing observations represented by `np.nan`

Use explicit double precision when constructing the matrix:

```python
x = np.asarray(x, dtype=np.float64)
```

The output is also `float64`. Avoid converting R and Python outputs independently to `float32` before comparison. Independent `float32` rounding can amplify an underlying difference near `1e-9` into a visible difference near `1.9e-6`.

At least two sample columns are required when fitting without a reference.

## Zero, negative, infinite, and nonnumeric values

The core `justvsn()` and `vsn_matrix()` functions do not automatically interpret zero as missing. Convert zeros before fitting when zero represents an undetected intensity:

```python
x = np.asarray(x, dtype=np.float64)
x[x == 0] = np.nan
```

For proteomics intensity matrices where all nonpositive values are considered missing:

```python
x[x <= 0] = np.nan
```

Convert nonfinite values to missing before fitting:

```python
x[~np.isfinite(x)] = np.nan
```

For pandas input containing text or mixed types:

```python
matrix = (
    data[intensity_columns]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float64, copy=True)
)
matrix[~np.isfinite(matrix)] = np.nan
matrix[matrix <= 0] = np.nan
```

Whether zero is missing is domain-specific. Do not replace genuine measured zeros unless that interpretation is appropriate for the assay.

## Missing rows and partially observed rows

Rows that are entirely missing across the selected samples are excluded from model fitting. Their positions are retained, and their transformed output remains entirely `np.nan`.

Rows that are only partially missing are retained in the input matrix. Every finite input value receives a final VSN-transformed value, while missing positions remain missing.

During robust LTS selection, a partially missing row normally receives a missing residual variance because residual variance intentionally propagates missing values. Such a row can be omitted from later trimmed fitting iterations, while the final fitted transformation is still applied to every finite value in that row.

## Minimum rows per stratum

The default is:

```python
min_data_points_per_stratum=42
```

When a stratum contains fewer than 42 rows, the implementation raises `ValueError` rather than issuing a warning. Lower the threshold only when fitting a deliberately small matrix:

```python
normalized = justvsn(
    x,
    min_data_points_per_stratum=10,
)
```

Setting the value to zero disables this safety check:

```python
normalized = justvsn(
    x,
    min_data_points_per_stratum=0,
)
```

The threshold counts matrix rows per stratum before entirely missing rows are removed from fitting. Passing the threshold therefore does not guarantee that the same number of usable rows remains after all-NA removal.

## Calibration modes

The default calibration mode is:

```python
calib="affine"
```

Affine calibration fits one offset and one log-scale per sample.

```python
normalized = justvsn(x, calib="affine")
```

No calibration fits one shared offset and one shared log-scale across the complete matrix:

```python
normalized = justvsn(x, calib="none")
```

Use `"none"` only when a shared transformation without sample-specific calibration is intentional.

## Transformation

For sample `j` and feature `i`, the fitted natural-scale transformation is:

```text
h_ij = asinh(exp(log_b_j) * x_ij + a_j)
```

The returned standard VSN output is:

```text
hx_ij = h_ij / log(2) - hoffset
```

where:

```text
hoffset = log2(2 * exp(mean(log_b)))
```

For affine calibration, each sample has its own offset `a_j` and log-scale `log_b_j`. For no calibration, one offset and one log-scale are shared by all samples.

The returned values are on the VSN log2-like scale. Raising `2` to the transformed values is an optional numerical back-transformation, but the result should not be interpreted as reconstructing the original raw intensity matrix.

## Model fitting

The implementation uses:

- profile maximum likelihood
- analytical gradients matching the R/C likelihood
- L-BFGS-B optimization through `scipy.optimize.minimize`
- robust least-trimmed-squares iterations
- five intensity slices per LTS iteration
- R-compatible quantile and missing-value behavior
- R-compatible starting parameters
- R-compatible final `hoffset`

### Default optimizer parameters

```python
DEFAULT_OPTIMPAR = {
    "factr": 5e7,
    "pgtol": 2e-4,
    "maxit": 60000,
    "trace": 0,
    "cvg_niter": 7,
    "cvg_eps": 0.0,
}
```

SciPy receives:

```text
maxcor = 5
ftol = factr * machine_epsilon
gtol = pgtol
maxiter = maxit
```

Offsets are unbounded. Log-scale parameters are bounded to `[-100, 100]`.

The public `trace` parameter is retained for API compatibility and progress reporting but is not forwarded as SciPy's former `iprint` option. `iprint` was removed from recent SciPy releases.

## Correct R-compatible LTS partitioning

The important compatibility correction is the implementation of:

```r
cut(rank(hmean, na.last = TRUE), breaks = 5)
```

When R receives a scalar number of breaks, R first creates equally spaced internal boundaries over the original rank range. R then expands only the first and last endpoints by 0.1% of that range.

The matching Python implementation is:

```python
cut_breaks = np.linspace(rank_min, rank_max, n_slices + 1)
cut_breaks[0] -= rank_span * 0.001
cut_breaks[-1] += rank_span * 0.001

slice_labels = (
    np.searchsorted(
        cut_breaks,
        rank_hmean,
        side="left",
    )
    - 1
)
slice_labels = np.clip(slice_labels, 0, n_slices - 1)
```

The earlier implementation expanded the complete range before generating all boundaries. That shifted every internal boundary and changed which sparse rows entered the LTS fit.

In the validated sparse dataset:

```text
Rows in input:                     7,816
Samples:                               7
Finite values compared:           18,023
Old Python RMSE versus R:       0.002289762308143096
Corrected Python RMSE versus R: 1.5963303695428504e-15
Corrected maximum error:        1.0658141036401503e-14
```

The corrected differences are at floating-point precision.

## Native R-versus-Python simulation validation

The corrected implementation was compared with native R `vsn::justvsn()` across 13 reproducible scenarios covering:

- 60 to 10,000 rows
- 4 to 24 samples
- dense and sparse matrices
- up to 75% random missingness
- up to 30% explicitly all-NA rows
- narrow and wide dynamic ranges
- high additive noise
- high multiplicative noise
- nearly multiplicative data
- strong outliers

Summary:

```text
Typical wide-range datasets:        about 1e-14 to 1e-13 maximum error
Sparse datasets:                    about 6e-14 maximum error
Dense narrow datasets:              about 1e-12 to 1e-9 maximum error
Nearly multiplicative data:         about 1.5e-9 maximum error
Deliberate strong-outlier scenario: about 1.11e-7 maximum error
Missing-value mismatches:           0 in every scenario
```

All tested scenarios passed an absolute tolerance of `1e-6`.

Recommended general validation:

```python
np.allclose(
    python_values,
    r_values,
    rtol=1e-7,
    atol=1e-6,
    equal_nan=True,
)
```

A stricter check is suitable for ordinary non-outlier datasets:

```python
np.allclose(
    python_values,
    r_values,
    rtol=1e-8,
    atol=1e-8,
    equal_nan=True,
)
```

Tiny nonzero differences are expected because native R and SciPy can evaluate floating-point reductions and terminate L-BFGS-B at numerically equivalent but not bit-identical points.

## Missing values during LTS selection

Row means are calculated from available transformed values.

Residual variance is calculated with missing values propagated, matching R:

```python
squared_residuals = (hy - hmean[:, np.newaxis]) ** 2
rvar = np.sum(squared_residuals, axis=1)
```

This intentionally does not use `np.nansum()`.

A partially missing row therefore has:

```text
finite transformed values
finite row mean
finite rank and intensity slice
NaN residual variance
```

Such a row is normally omitted from later trimmed fitting iterations. Rows in the lowest-intensity slice are retained by the explicit R-compatible selection rule. Regardless of LTS selection, the final fitted transformation is applied to every finite input value.

## API

### `justvsn()`

```python
justvsn(
    x,
    reference=None,
    strata=None,
    lts_quantile=0.9,
    subsample=0,
    verbose=False,
    calib="affine",
    pstart=None,
    min_data_points_per_stratum=42,
    optimpar=None,
    defaultpar=None,
)
```

Returns the transformed `float64` NumPy matrix directly.

### `vsn_matrix()`

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

Parameters:

- `x`: two-dimensional NumPy array with features in rows and samples in columns
- `reference`: optional fitted `VsnResult` used for reference normalization
- `strata`: optional one-based integer labels for row strata, covering consecutive values from 1 through the number of strata
- `lts_quantile`: retained LTS fraction, between `0.5` and `1.0`
- `subsample`: number of rows sampled per stratum; `0` uses all rows
- `verbose`: print fitting diagnostics
- `return_data`: calculate and store the transformed matrix in `result.hx`
- `calib`: `"affine"` or `"none"`
- `pstart`: optional custom starting coefficients
- `min_data_points_per_stratum`: minimum number of rows required per stratum
- `optimpar`: overrides selected optimizer settings
- `defaultpar`: overrides the default optimizer dictionary

### `VsnResult`

The result contains:

- `hx`: transformed matrix when `return_data=True`
- `coefficients`: fitted offsets and log-scales
- `mu`: transformed row means
- `sigsq`: estimated residual variance
- `hoffset`: final stratum-specific VSN offset
- `strata`: row stratum labels
- `lbfgsb`: optimizer status code; zero indicates success
- `calib`: calibration mode

Coefficient layout:

```python
offsets = result.coefficients[:, :, 0]
log_scales = result.coefficients[:, :, 1]
scales = np.exp(log_scales)
```

### Direct transformation

```python
import numpy as np
from vsn import vsn2_trsf

transformed = vsn2_trsf(
    x=x,
    p=result.coefficients,
    strata=np.ones(x.shape[0], dtype=int),
    hoffset=result.hoffset,
    calib="affine",
)
```

## Reference normalization

A fitted `VsnResult` can be supplied as a reference:

```python
reference_fit = vsn_matrix(reference_matrix)
normalized_to_reference = justvsn(
    target_matrix,
    reference=reference_fit,
)
```

The target matrix must have the same number of rows as the fitted reference, and rows must represent the same features in the same order.

## Strata

Optional strata are one-based integer labels, one label per row:

```python
strata = np.array([1, 1, 1, 2, 2, 2], dtype=int)
result = vsn_matrix(
    x,
    strata=strata,
    min_data_points_per_stratum=0,
)
```

Labels must cover consecutive values from `1` through the number of strata. Calibration mode `"none"` does not support multiple strata.

## Subsampling

For very large matrices, fitting can use a fixed number of rows per stratum:

```python
result = vsn_matrix(
    x,
    subsample=10000,
)
```

The final fitted transformation is applied to the complete matrix. When reproducibility across separate runs matters, record the package versions, input ordering, and any random-state behavior used by the surrounding workflow.

## Optimizer overrides

```python
from vsn import vsn_matrix

result = vsn_matrix(
    x,
    optimpar={
        "factr": 5e7,
        "pgtol": 2e-4,
        "maxit": 60000,
        "trace": 0,
        "cvg_niter": 7,
        "cvg_eps": 0.0,
    },
)
```

Python uses underscores in `cvg_niter` and `cvg_eps`, corresponding to R's `cvg.niter` and `cvg.eps`.

Tightening optimizer tolerances does not guarantee a closer match to R. Native R and SciPy may terminate at slightly different points on a flat numerical optimum.

## Checking convergence

For a full fit:

```python
result = vsn_matrix(x)

if result.lbfgsb != 0:
    raise RuntimeError(
        f"VSN optimizer returned status {result.lbfgsb}"
    )
```

Always inspect mean-SD behavior and sample distributions after normalization. Successful optimizer termination does not replace data-quality checks.

## Matrix runner

Run VSN on explicitly named columns:

```bash
python run.py input.tsv output.tsv \
  --id-columns "Protein.Group,Protein.Names,Genes" \
  --intensity-columns "Sample1,Sample2,Sample3,Sample4"
```

Select intensity columns with regular expressions:

```bash
python run.py input.tsv output.tsv \
  --id-columns "Protein.Group,Protein.Names,Genes" \
  --intensity-regex "^F:"
```

The runner:

- reads CSV or TSV input
- preserves requested identifier columns
- converts nonnumeric and nonfinite intensity entries to missing values
- treats zero as missing by default
- preserves finite values in partially missing rows
- writes VSN-transformed values
- writes fitted parameters to a separate CSV file

Use `--keep-zero` only when zero is a genuine measured intensity.

## Marimo dashboard

Hosted at [marimo-wasm](https://molab.marimo.io/github/animesh/vsn/blob/master/vsn2mo.py/wasm), to run locally:

```bash
uv run marimo run vsn2mo.py
```

The standalone dashboard supports:

- CSV and TSV uploads
- intensity-column prefix selection
- affine and no-calibration modes
- editable VSN and optimizer parameters
- minimum-row validation
- raw and transformed diagnostics
- sample regression views
- timestamp-safe result columns
- parameterized download filenames

The dashboard contains its own implementation and does not import the installed `vsn` module. The embedded affine and no-calibration paths were checked against `vsn.py` at floating-point precision.

The dashboard also creates optional `pow2_` columns. These are numerical back-transformations of the VSN-scale values, not reconstructed raw intensities. Use them only when a downstream workflow explicitly expects values exponentiated from the VSN log2-like scale. See the [basic differential-expression workflow](https://fuzzylife.substack.com/p/basic-differential-expression-analysis) for the specific context in which these columns are used.

## Quarto Shiny dashboard

Hosted at [posit-cloud](https://fuzzylife-vsn.share.connect.posit.cloud/), to run locally:

```bash
quarto preview run.qmd
```

The dashboard uses the R `vsn` package directly and provides raw and transformed mean-SD plots, distributions, sample boxplots, regression diagnostics, timestamp-safe result columns, and parameterized downloads. More details in https://fuzzylife.substack.com/p/variance-stabilizing-normalization 

## R reference workflow

The equivalent R normalization is:

```r
library(vsn)

x <- as.matrix(data[, intensity_columns, drop = FALSE])
x[x == 0] <- NA_real_

normalized <- vsn::justvsn(x)
```

For an intentionally small dataset:

```r
normalized <- vsn::justvsn(
    x,
    minDataPointsPerStratum = 0
)
```

## Building and testing the package

Build the source distribution and wheel:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Windows PowerShell:

```powershell
python -m venv .venv-wheel-test
.\.venv-wheel-test\Scripts\python.exe -m pip install dist\vsn-0.1.1-py3-none-any.whl
.\.venv-wheel-test\Scripts\python.exe -W error -c "from vsn import justvsn; print('Wheel import passed')"
```

## Troubleshooting

### Import still resolves to a local checkout

Check the imported path:

```bash
python -c "import vsn; print(vsn.__file__)"
```

If the path points to a development checkout, uninstall the editable package before testing the PyPI wheel:

```bash
python -m pip uninstall vsn
python -m pip install --upgrade vsn
```

### Matrix has fewer than 42 rows

Lower `min_data_points_per_stratum` only when the small fit is intentional. Setting it to zero disables the safety check.

### Text values cause conversion errors

Coerce selected pandas columns with `pd.to_numeric(errors="coerce")` before converting to NumPy.

### R and Python are not exactly equal

Compare `float64` values with a documented tolerance. Verify that both fits use the same rows, samples, order, missing-value handling, and parameters. Do not compare independently rounded `float32` matrices.

### Optimizer does not converge

Inspect `result.lbfgsb`, enable progress reporting with `trace`, check the number of usable observations, examine the dynamic range, and avoid assuming that stricter tolerances will necessarily improve agreement with R.

## Release notes

### 0.1.1

- corrected package documentation to use `vsn.py` and `from vsn import ...`
- documented `justvsn()` as the primary convenience API
- documented native R validation and simulation tools
- documented missing-value, calibration, reference, strata, and optimizer behavior
- documented SciPy compatibility without the removed `iprint` option

### 0.1.0

- initial PyPI release

## Current validation status


The following components have been independently checked in the current implementation:

- profile negative log-likelihood
- analytical gradient
- affine parameterization
- no-calibration shared parameterization
- R-compatible starting parameters
- R-compatible `hoffset`
- L-BFGS-B memory setting
- missing-value propagation during LTS residual selection
- R-compatible quantile interpolation
- R-compatible five-slice rank partitioning
- restoration of all-missing rows in the final output
- application of the final transformation to partially observed rows
- `justvsn()` wrapper equality with `vsn_matrix(...).hx`
- native R-versus-Python simulation agreement across 13 scenarios
- standalone Marimo affine and no-calibration agreement with `vsn.py`
- clean wheel installation and import
- execution with Python 3.14, NumPy, and SciPy

## References

Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. Variance stabilization applied to microarray data calibration and to the quantification of differential expression. Bioinformatics. 2002;18(Suppl 1):S96-S104.

Bioconductor package: `vsn`, by Wolfgang Huber and contributors.

## Citation and attribution

If this package is used in scientific work:

1. Cite the  above mentioned original VSN publication.
2. Cite or link to the [Python repository](https://github.com/animesh/vsn).
3. Record the installed `vsn` [package](https://pypi.org/project/vsn/) version in the analysis environment.

The installed version can be retrieved with:

```python
import platform
import numpy as np
import scipy
from importlib.metadata import version
print("vsn:", version("vsn"))
print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
```

