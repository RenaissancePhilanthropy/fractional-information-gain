# Changelog

All notable changes to this project will be documented in this page.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] -- Initial Release

### Added

- FIG-V (Validation) and FIG-C (Confidence) metrics for binary response tasks.
- NumPy reference implementation (`fig.fig`).
- Differentiable PyTorch implementation (`fig.fig_torch`) with loss wrappers.
- R package (`figmetric`) mirroring the Python API.
- Empirical Bayes shrinkage for per-item baseline estimation.
- Optional calibration checking (ECE) for FIG-V.
