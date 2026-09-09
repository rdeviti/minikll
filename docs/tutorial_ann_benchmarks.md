# ANN-Benchmarks Distance Tutorial

This tutorial shows how to use `minikll` with a standard ANN-Benchmarks HDF5
dataset. ANN-Benchmarks datasets are distributed as HDF5 files with pre-split
`train`/`test` vectors and top-100 ground truth neighbor data, including a
`distances` matrix.

The example estimates the distribution of the true `rank`-nearest-neighbor
distance across all queries. For example, on SIFT Euclidean data, `--rank 10`
summarizes the distribution of each query's exact 10th-nearest-neighbor
distance. This can be useful for setting radius thresholds, inspecting dataset
difficulty, or comparing compact KLL summaries against exact quantiles.

## Prerequisites

- A standard ANN-Benchmarks `.hdf5` dataset, such as SIFT Euclidean.
- HDF5 with the C++ API installed and discoverable by CMake.
- `minikll` built with examples enabled.

ANN-Benchmarks distributes ready-made datasets as HDF5 files. For example, to
download SIFT Euclidean (~525 MB):

```bash
curl -O https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

Other datasets follow the same `https://ann-benchmarks.com/<name>.hdf5` pattern;
see the [ANN-Benchmarks project](https://github.com/erikbern/ann-benchmarks) for
the full list. The tutorial does not assume any repository layout — pass the
dataset path on the command line.

## Build

```bash
cmake -S /path/to/minikll -B /tmp/minikll_examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIKLL_BUILD_TESTS=OFF \
  -DMINIKLL_BUILD_EXAMPLES=ON
cmake --build /tmp/minikll_examples
```

For editors such as VSCode, configure with a compilation database if you want
IntelliSense to pick up the same include paths as the compiler:

```bash
cmake -S /path/to/minikll -B /tmp/minikll_examples \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMINIKLL_BUILD_TESTS=OFF \
  -DMINIKLL_BUILD_EXAMPLES=ON
```

Then point the C/C++ extension at:

```text
/tmp/minikll_examples/compile_commands.json
```

If CMake cannot find HDF5 automatically, pass your HDF5 prefix in the usual
CMake way, for example:

```bash
cmake -S /path/to/minikll -B /tmp/minikll_examples \
  -DMINIKLL_BUILD_EXAMPLES=ON \
  -DHDF5_ROOT=/path/to/hdf5/prefix
```

## Run

```bash
/tmp/minikll_examples/ann_benchmarks_distances \
  --dataset /path/to/sift-128-euclidean.hdf5 \
  --rank 10 \
  --kll-k 200 \
  --fixed-size 256
```

The output is CSV-like:

```text
dataset=/path/to/sift-128-euclidean.hdf5
distances_dataset=distances
rank=10
queries=10000
mode=random
odd_policy=carry
kll_k=200
estimated_rank_error=0.0132948
active_retained=...
fixed_retained=...
batch_fixed_retained=...

q,exact,active_kll,fixed_kll,batch_fixed_kll
0.5,...
0.9,...
0.95,...
0.99,...
```

`active_kll` is produced after `update_batch(...)` by `finalize(false)`, which
does not add a final compaction pass. `fixed_kll` is produced by
`finalize(true)`, which compacts once to the requested fixed-size target.
`batch_fixed_kll` uses `WeightedSketch1D::from_sorted_values(...)` directly
after sorting the distance column once. That direct path is useful when the
whole column is already materialized and you want a compact fixed-size summary.

The example also queries all displayed percentiles with `quantiles(qs)`, so the
retained sketch is scanned once for the whole percentile list rather than once
per percentile.

## Odd-Item Policy

The default `--odd-policy carry` preserves represented weight when a compaction
pass has an unpaired item.

The legacy `--odd-policy drop_largest` mode is explicit:

```bash
/tmp/minikll_examples/ann_benchmarks_distances \
  --dataset /path/to/sift-128-euclidean.hdf5 \
  --rank 10 \
  --odd-policy drop_largest
```

`drop_largest` makes each compaction pass a uniform truncate-to-even plus
pairwise transform. That can simplify fixed batches, vectorized pipelines, or
reproducing older outputs, but it drops represented weight and is therefore not
the default.

## Source

The full tutorial program is in:

```text
minikll/examples/ann_benchmarks_distances.cpp
```
