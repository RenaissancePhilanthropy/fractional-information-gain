# R Package

The `figmetric` R package provides the same FIG metrics as the Python library, with an
identical API and numerically equivalent results. It lives in the `r/` subdirectory of
the repository.

## Installation

The R package is not on CRAN. Install directly from the repository:

```r
# Using devtools
devtools::install_github(
  "RenaissancePhilanthropy/fractional-information-gain",
  subdir = "r"
)
```

Or load it in-place for development:

```r
pkgload::load_all("r/")
```

## Dependencies

- R (any recent version)
- No external R package dependencies for the core functions

For running tests: `testthat`, `pkgload`, `jsonlite`.

## Usage

The R API mirrors the Python API exactly -- same parameter names, same return structure:

```r
library(figmetric)

results <- fractional_information_gain_validation(
  y_pred_eval = c(0.8, 0.6, 0.9, 0.5, 0.7),
  y_eval = c(1, 0, 1, 0, 1),
  item_id_eval = c(1, 1, 2, 2, 1),
  student_id_eval = c(1, 1, 2, 2, 3),
  y_train = c(1, 1, 0, 1, 0),
  item_id_train = c(1, 1, 2, 2, 2)
)

cat("FIG-V:", results$fig_v, "\n")
cat("FIG-V pooled:", results$fig_v_pooled, "\n")
```

FIG-C works similarly:

```r
results <- fractional_information_gain_confidence(
  y_pred_eval = c(0.8, 0.6, 0.9, 0.5, 0.7),
  item_id_eval = c(1, 1, 2, 2, 1),
  student_id_eval = c(1, 1, 2, 2, 3),
  y_train = c(1, 1, 0, 1, 0),
  item_id_train = c(1, 1, 2, 2, 2)
)

cat("FIG-C:", results$fig_c, "\n")
```

## Cross-Language Parity

The Python test suite includes cross-language parity tests
(`tests/test_cross_language.py`) that verify the Python NumPy, Python PyTorch, and R
implementations produce numerically equivalent results within floating-point tolerance.

## Running R Tests

From the `r/` subdirectory:

```bash
cd r/
Rscript -e "pkgload::load_all('.'); testthat::test_dir('tests/testthat')"
```
