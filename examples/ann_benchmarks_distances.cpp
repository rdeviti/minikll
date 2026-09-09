/**
 * @file ann_benchmarks_distances.cpp
 *
 * Tutorial executable for applying minikll to an ANN-Benchmarks HDF5 file.
 *
 * ANN-Benchmarks datasets usually contain a `distances` matrix whose rows are
 * query vectors and whose columns are exact-neighbor ranks. This example reads
 * one rank column, sketches its distribution, and prints exact versus KLL
 * quantiles.
 */

#include <minikll/minikll.hpp>

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/** Command-line options for the tutorial executable. */
struct Options {
  std::string dataset_path;
  std::string distances_dataset = "distances";
  std::size_t rank = 10;
  std::size_t max_queries = 0;
  std::size_t stream_k = 200;
  std::size_t fixed_size = 256;
  std::uint64_t seed = 0;
  minikll::CompactMode mode = minikll::CompactMode::Random;
  minikll::OddPolicy odd_policy = minikll::OddPolicy::Carry;
};

/** Print CLI usage without depending on any command-line parsing library. */
void print_usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --dataset PATH [options]\n\n"
      << "Options:\n"
      << "  --dataset PATH             ANN-Benchmarks HDF5 file.\n"
      << "  --distances NAME           HDF5 dataset name (default: distances).\n"
      << "  --rank N                   1-based neighbor rank to summarize (default: 10).\n"
      << "  --max-queries N            Limit number of query rows (default: all).\n"
      << "  --kll-k N                  Streaming KLL level capacity (default: 200).\n"
      << "  --fixed-size N             Final compacted retained target (default: 256).\n"
      << "  --seed N                   Random compaction seed (default: 0).\n"
      << "  --mode random|deterministic\n"
      << "  --odd-policy carry|drop_largest\n";
}

/** Parse a non-negative integer option with full-string validation. */
std::size_t parse_size(const std::string& value, const std::string& name) {
  std::size_t parsed = 0;
  std::size_t consumed = 0;
  try {
    parsed = static_cast<std::size_t>(std::stoull(value, &consumed));
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid value for " + name + ": " + value);
  }
  if (consumed != value.size()) {
    throw std::invalid_argument("invalid value for " + name + ": " + value);
  }
  return parsed;
}

/** Parse an unsigned 64-bit integer option, currently used for the RNG seed. */
std::uint64_t parse_u64(const std::string& value, const std::string& name) {
  return static_cast<std::uint64_t>(parse_size(value, name));
}

/**
 * Parse the tutorial's flags and validate values that would otherwise produce
 * ambiguous output, such as rank 0 or a zero retained-size target.
 */
Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--dataset") {
      opts.dataset_path = require_value(arg);
    } else if (arg == "--distances") {
      opts.distances_dataset = require_value(arg);
    } else if (arg == "--rank") {
      opts.rank = parse_size(require_value(arg), arg);
    } else if (arg == "--max-queries") {
      opts.max_queries = parse_size(require_value(arg), arg);
    } else if (arg == "--kll-k") {
      opts.stream_k = parse_size(require_value(arg), arg);
    } else if (arg == "--fixed-size") {
      opts.fixed_size = parse_size(require_value(arg), arg);
    } else if (arg == "--seed") {
      opts.seed = parse_u64(require_value(arg), arg);
    } else if (arg == "--mode") {
      opts.mode = minikll::compact_mode_from_string(require_value(arg));
    } else if (arg == "--odd-policy") {
      opts.odd_policy = minikll::odd_policy_from_string(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }

  if (opts.dataset_path.empty()) {
    throw std::invalid_argument("--dataset is required");
  }
  if (opts.rank == 0) {
    throw std::invalid_argument("--rank is 1-based and must be positive");
  }
  if (opts.stream_k == 0) {
    throw std::invalid_argument("--kll-k must be positive");
  }
  if (opts.fixed_size == 0) {
    throw std::invalid_argument("--fixed-size must be positive");
  }
  return opts;
}

/**
 * Read one 1-based neighbor-rank column from an ANN-Benchmarks distances matrix.
 *
 * The function uses an HDF5 hyperslab so it reads only the requested rank column
 * rather than materializing the full `queries x neighbors` matrix.
 */
std::vector<double> read_rank_distances(const Options& opts) {
  H5::H5File file(opts.dataset_path, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(opts.distances_dataset);
  H5::DataSpace space = dataset.getSpace();

  const int ndims = space.getSimpleExtentNdims();
  if (ndims != 2) {
    throw std::runtime_error("expected a 2D distances dataset");
  }

  hsize_t dims[2] = {0, 0};
  space.getSimpleExtentDims(dims);
  const std::size_t n_queries = static_cast<std::size_t>(dims[0]);
  const std::size_t n_neighbors = static_cast<std::size_t>(dims[1]);
  if (opts.rank > n_neighbors) {
    throw std::runtime_error(
        "--rank exceeds distances dataset neighbor count");
  }

  const std::size_t rows =
      opts.max_queries == 0 ? n_queries : std::min(opts.max_queries, n_queries);
  const std::size_t rank_col = opts.rank - 1;

  // Select all requested query rows and the single target rank column.
  hsize_t offset[2] = {0, rank_col};
  hsize_t count[2] = {rows, 1};
  space.selectHyperslab(H5S_SELECT_SET, count, offset);

  hsize_t mem_dims[1] = {rows};
  H5::DataSpace mem_space(1, mem_dims);
  std::vector<double> distances(rows);
  dataset.read(
      distances.data(), H5::PredType::NATIVE_DOUBLE, mem_space, space);

  return distances;
}

/**
 * Exact inclusive quantile used as a baseline for the tutorial output.
 *
 * This intentionally mirrors minikll's quantile convention: target cumulative
 * rank is ceil(q * n), with q=0 returning the first sorted item.
 */
double exact_quantile_from_sorted(
    const std::vector<double>& sorted_values,
    double q) {
  if (sorted_values.empty()) {
    throw std::invalid_argument("exact quantile undefined for empty input");
  }
  if (!(q >= 0.0 && q <= 1.0)) {
    throw std::invalid_argument("q must be in [0, 1]");
  }
  std::size_t target = 0;
  if (q > 0.0) {
    target = static_cast<std::size_t>(
        std::ceil(q * static_cast<double>(sorted_values.size())));
  }
  if (target == 0) {
    return sorted_values.front();
  }
  return sorted_values[
      std::min(target - 1, sorted_values.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_args(argc, argv);
    std::vector<double> distances = read_rank_distances(opts);

    // Sort once for exact baselines and for the direct fixed-size sketch path.
    std::vector<double> distances_sorted = distances;
    std::sort(distances_sorted.begin(), distances_sorted.end());

    // Build a level-based sketch over the selected true-neighbor distance
    // column. `update_batch` is the right path when the column is already
    // materialized; for true online streams, call `update` as values arrive.
    minikll::KLLSketch1D sketch(
        opts.stream_k,
        opts.fixed_size,
        opts.mode,
        opts.seed,
        true,
        opts.odd_policy);
    sketch.update_batch(std::move(distances));

    // Compare the natural retained sketch, final compacted sketch, and direct
    // sorted-column constructor. The direct path avoids the level buffer and is
    // useful when you want a fixed-size summary for a full column.
    const auto active = sketch.finalize(false);
    const auto fixed = sketch.finalize(true);
    const auto batch_fixed = minikll::WeightedSketch1D::from_sorted_values(
        distances_sorted,
        opts.fixed_size,
        opts.mode,
        opts.seed,
        opts.odd_policy);
    const std::vector<double> qs = {0.5, 0.9, 0.95, 0.99};
    const auto active_qs = active.quantiles(qs);
    const auto fixed_qs = fixed.quantiles(qs);
    const auto batch_fixed_qs = batch_fixed.quantiles(qs);

    std::cout << "dataset=" << opts.dataset_path << "\n";
    std::cout << "distances_dataset=" << opts.distances_dataset << "\n";
    std::cout << "rank=" << opts.rank << "\n";
    std::cout << "queries=" << distances_sorted.size() << "\n";
    std::cout << "mode=" << minikll::compact_mode_to_string(opts.mode) << "\n";
    std::cout << "odd_policy="
              << minikll::odd_policy_to_string(opts.odd_policy) << "\n";
    std::cout << "kll_k=" << opts.stream_k << "\n";
    std::cout << "estimated_rank_error="
              << minikll::normalized_rank_error(opts.stream_k) << "\n";
    std::cout << "active_retained=" << active.retained() << "\n";
    std::cout << "fixed_retained=" << fixed.retained() << "\n";
    std::cout << "batch_fixed_retained=" << batch_fixed.retained() << "\n\n";

    std::cout << "q,exact,active_kll,fixed_kll,batch_fixed_kll\n";
    for (std::size_t i = 0; i < qs.size(); ++i) {
      std::cout << qs[i] << ","
                << exact_quantile_from_sorted(distances_sorted, qs[i]) << ","
                << active_qs[i] << "," << fixed_qs[i] << ","
                << batch_fixed_qs[i] << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n\n";
    print_usage(argv[0]);
    return 1;
  }
}
