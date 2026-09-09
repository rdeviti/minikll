# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

Initial public release.

- Header-only C++17 KLL-like weighted quantile sketch with a minimal
  `(values, weights)` data model and optional fixed-size `(+inf, 0)` padding.
- `WeightedSketch1D`, streaming `KLLSketch1D`, and column-wise `KLLSketchND`.
- Deterministic and seeded-random compaction; mass-preserving `Carry` and legacy
  `DropLargest` odd-item policies.
- Single- and batch-quantile queries, weighted-median and sorted-array merge
  utilities, and DataSketches-style rank-error helpers.
- Optional pybind11 Python bindings and an HDF5 ANN-Benchmarks example, both off
  by default so the C++ core stays dependency-free.
