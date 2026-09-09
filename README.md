<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.svg">
    <img src="docs/logo.svg" alt="minikll — weighted quantile sketches" width="460">
  </picture>
</p>

[![CI](https://github.com/rdeviti/minikll/actions/workflows/ci.yml/badge.svg)](https://github.com/rdeviti/minikll/actions/workflows/ci.yml)

# minikll

`minikll` is a small, standalone C++17 KLL-like quantile sketch library.

The core API uses a simple representation:

- `values`: sorted ascending `double` values
- `weights`: non-negative integer weights
- optional fixed-size padding: `(value=+inf, weight=0)` only when fixed-size output is
  useful for your application

Outside fixed-shape protocols, padding is usually unnecessary. By default,
`to_arrays()` returns only active retained items. Use `to_arrays(fixed_size=true)` or
`to_padded_arrays()` if you need exactly `s` entries. If a sketch currently
retains more than `s` active items, use a fixed-size finalize or merge path
before requesting fixed-size arrays.

## Relationship To Apache DataSketches

Apache DataSketches has a mature, Apache-2.0 licensed KLL implementation:

- <https://github.com/apache/datasketches-cpp>
- <https://datasketches.apache.org/docs/KLL/KLLSketch.html>

Use Apache DataSketches when you need a production sketch family with its
serialization format, broad language ecosystem, and formally documented error
profiles.

`minikll` is intentionally narrower. It emphasizes:

- **Minimal core**: header-only C++17 implementation with no runtime dependency
  in the C++ API.
- **Plain weighted arrays**: direct import/export of `(values, weights)` instead
  of a sketch-specific binary format.
- **Weighted-array utilities**: weighted median, two-way merge, and exact
  heap-based K-way merge for already sorted weighted summaries.
- **Optional fixed-size export**: `(+inf, 0)` padding is available only when your
  use case needs fixed-shape arrays.
- **Random compaction by default**: seeded random compaction is the default;
  deterministic keep-right compaction is available for reproducible diagnostics.
- **Mass-preserving odd compaction**: odd compaction carries one boundary item
  forward unchanged instead of dropping it.
- **Legacy uniform odd compaction**: `OddPolicy::DropLargest` is available when
  you need each compaction pass to be a uniform truncate-to-even plus pairwise
  transform, which can simplify fixed batches, vectorized pipelines, or
  reproducibility against older outputs. It drops represented weight, so it is
  not the default.
- **Optional final compaction**: streaming sketches finalize to all retained
  items by default; pass `fixed_size=true` to compact once to `s` retained items.
- **ANN-benchmarks-friendly data flow**: constructors consume standard sorted
  columns, unsorted columns, and row-major matrices, which makes it easy to feed
  distance or score columns produced by ANN benchmark pipelines without adopting
  a sketch serialization layer.
- **Batch-oriented queries and updates**: `quantiles(qs)` answers many
  percentile queries with one retained-item scan, and `update_batch(...)`
  amortizes sorting/compaction when a full vector is already materialized.
- **Small Python binding**: optional pybind11 bindings expose the same simple
  arrays to Python.

For exact, non-compacting examples, this library's inclusive rank rule matches
Apache DataSketches when Apache is queried with `inclusive=true`. Apache's
default Python `get_quantile(rank)` uses `inclusive=false`, which differs for
even-sized inputs at boundary ranks such as the median of `[1, 2]`.

For larger streams, item-for-item outputs are not expected to match Apache.
Evaluate approximate sketches by rank error, not exact returned item equality.

## Build C++

The core C++ library is header-only and has no runtime dependencies beyond a
C++17 compiler and the standard library.

```bash
cmake -S . -B /tmp/minikll_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/minikll_build
ctest --test-dir /tmp/minikll_build --output-on-failure
```

## Dependencies

| Component | Default | Dependencies |
| --- | --- | --- |
| Core C++ headers | always available | C++17 compiler |
| C++ tests | on | CMake, C++17 compiler |
| Python bindings | off | Python development headers, pybind11 |
| ANN-Benchmarks example | off | HDF5 C++ library |

Optional components are disabled unless explicitly requested:

```bash
-DMINIKLL_BUILD_PYTHON=ON
-DMINIKLL_BUILD_EXAMPLES=ON
```

This keeps the default package easy to vendor: downstream C++ users can include
`minikll/minikll.hpp` without installing Python, pybind11, or HDF5.

Common optional dependency installs:

```bash
# macOS with Homebrew
brew install cmake hdf5
python3 -m pip install pybind11

# Conda
conda install -c conda-forge cmake hdf5 pybind11

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y cmake g++ libhdf5-dev pybind11-dev python3-dev
```

Useful upstream links:

- CMake: <https://cmake.org/download/>
- pybind11: <https://pybind11.readthedocs.io/>
- HDF5: <https://www.hdfgroup.org/downloads/hdf5/>
- ANN-Benchmarks datasets/project: <https://github.com/erikbern/ann-benchmarks>

## Tutorials

- [ANN-Benchmarks distance tutorial](docs/tutorial_ann_benchmarks.md): builds a
  KLL sketch over the ground-truth `distances` matrix in a standard
  ANN-Benchmarks HDF5 dataset.

## Minimal C++ Example

```cpp
#include <minikll/minikll.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  std::vector<double> values = {3.0, 1.0, 2.0, 5.0, 4.0};
  std::sort(values.begin(), values.end());

  auto sketch = minikll::WeightedSketch1D::from_sorted_values(values, 4);
  std::cout << sketch.median() << "\n";
  auto qs = sketch.quantiles({0.5, 0.9, 0.99});
  std::cout << qs[1] << "\n";

  auto active = sketch.to_arrays();        // active retained entries only
  auto padded = sketch.to_padded_arrays(); // exactly 4 entries, padded if needed
}
```

## Python Bindings

Python bindings are optional and use pybind11.

```bash
python3 -m pip install pybind11
cmake -S . -B /tmp/minikll_py_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIKLL_BUILD_TESTS=OFF \
  -DMINIKLL_BUILD_PYTHON=ON
cmake --build /tmp/minikll_py_build
```

With the built module on your `PYTHONPATH` (for example
`PYTHONPATH=/tmp/minikll_py_build`):

```python
import minikll as mk

# Deterministic mode keeps the right item of each pair, so output is stable.
sk = mk.WeightedSketch1D.from_sorted_values(
    [1.0, 2.0, 3.0, 4.0], s=2, mode=mk.CompactMode.Deterministic)
print(sk.median())                    # 2.0
print(sk.quantiles([0.5, 0.9]))       # [2.0, 4.0]
print(sk.to_arrays())                 # ([2.0, 4.0], [2, 2])
print(sk.to_arrays(fixed_size=True))  # same here because retained == s

stream = mk.KLLSketch1D(k=64, s=128)
stream.update_batch([3.0, 1.0, 2.0])
print(stream.finalize().to_arrays())             # no extra final compaction
print(stream.finalize(fixed_size=True).to_arrays(fixed_size=True))
```

The CMake target is disabled by default so the C++ library remains dependency
free. Enable it with `-DMINIKLL_BUILD_PYTHON=ON` after installing pybind11.

For packaging, the usual best practice is to keep bindings as an optional build
target or a separate wheel/package. That avoids making Python tooling a required
dependency for C++ consumers.

## Main C++ APIs

- `WeightedSketch1D::from_samples(samples, s, mode, seed, odd_policy)`
- `WeightedSketch1D::from_sorted_values(values_sorted, s, mode, seed, odd_policy)`
- `WeightedSketch1D::from_retained_samples(samples, size_limit)`
- `WeightedSketch1D::from_arrays(values, weights, s)`
- `WeightedSketch1D::to_arrays(fixed_size=false)`
- `WeightedSketch1D::to_padded_arrays()`
- `WeightedSketch1D::quantile(q)`, `quantiles(qs)`, and `median()`
- `WeightedSketch1D::merge(other, mode, seed, fixed_size=false, odd_policy)`
- `KLLSketch1D(k, s, mode, seed, ignore_nan, odd_policy)`
- `KLLSketch1D::update(x)`, `update_many(xs)`, `update_batch(xs, pre_sorted=false)`,
  and `update_sorted_batch(xs)`
- `KLLSketch1D::finalize(fixed_size=false)`
- `KLLSketchND(k, s, d, mode, seed, ignore_nan, odd_policy)`
- `build_sketch_from_sorted_values(..., fixed_size=false, odd_policy)`
- `build_sketch_from_column(..., fixed_size=false, odd_policy)`
- `merge_two_sorted_weighted(...)`
- `merge_k_sorted_lists(...)`
- `merge_weighted_sketches(..., fixed_size=false, odd_policy)`
- `weighted_median_from_sorted_weighted(...)`
- `weighted_median_from_k_sorted_sketches(...)`
- `normalized_rank_error(k, pmf=false)` and `suggest_k(epsilon, pmf=false)`

## Notes

- Random compaction is seeded and reproducible within this library, but no
  cross-language RNG bitstream compatibility is promised.
- Quantile queries accept any rank `q` in `[0, 1]`; a percentile `p` is queried
  as `q = p / 100.0`. Use `quantiles(qs)` for many percentiles at once.
- Deterministic compaction keeps the right item from each adjacent pair.
- `OddPolicy::Carry` preserves represented weight. `OddPolicy::DropLargest`
  removes the largest unpaired item before pairwise compaction.
- Equal retained values are combined by summing their weights. This preserves
  the weighted empirical distribution exactly.
- `from_samples` rejects `NaN` values because NaNs do not have a strict sort
  order.
- Inputs that claim to be sorted are checked and sorted defensively if needed.
- `update_batch` sorts defensively and compacts after appending the full batch.
  Its output is a valid sketch but not necessarily bit-identical to one-by-one
  streaming updates, because compaction boundaries can differ.
- `merge_weighted_sketches` collects all active retained items and compacts at
  most once. `merge_k_sorted_lists` remains as an exact array-level utility.
- The rank-error helpers use the same empirical KLL fit popularized by
  DataSketches; treat them as estimates for this minimal implementation.
- This folder is MIT licensed. See `LICENSE`.
