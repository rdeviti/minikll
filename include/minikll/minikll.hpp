#ifndef MINIKLL_MINIKLL_HPP_
#define MINIKLL_MINIKLL_HPP_

/**
 * @file minikll.hpp
 *
 * A dependency-free C++17 KLL-like quantile sketch library.
 *
 * The core data model is intentionally small: sorted retained values plus
 * non-negative integer weights. Callers can export active arrays by default, or
 * request fixed-size padded arrays with (+inf, 0) entries when a downstream
 * protocol needs a fixed shape.
 *
 * This is not a byte-compatible Apache DataSketches implementation. It is a
 * compact sketch with deterministic and seeded-random compaction modes, useful
 * when callers want a minimal data layout instead of a production sketch
 * serialization format.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minikll {

using weight_type = std::uint64_t;

/**
 * Compaction policy used when a retained sorted list exceeds its size limit.
 *
 * Deterministic mode always keeps the right item of each adjacent pair. Random
 * mode chooses the kept side once per compaction pass from a seeded RNG.
 */
enum class CompactMode {
  Deterministic,
  Random,
};

/**
 * Odd-item handling for compaction passes.
 *
 * Carry preserves total represented weight and is the default for standalone
 * sketch accuracy. DropLargest is a legacy/uniform-transform option: each pass
 * first truncates to an even length, then applies the same pairwise transform to
 * every remaining item. That shape can be convenient for fixed batches,
 * vectorized pipelines, or reproducing older outputs, but it drops mass.
 */
enum class OddPolicy {
  Carry,
  DropLargest,
};

/**
 * One retained item in the weighted sketch representation.
 *
 * `weight` is the number of original stream items represented by `value`.
 */
struct WeightedSample {
  double value = 0.0;
  weight_type weight = 0;
};

/**
 * Array export representation.
 *
 * Active entries have positive weights. If fixed-size export is requested,
 * padding entries use value `+inf` and weight `0`.
 */
struct SketchArrays {
  std::vector<double> values;
  std::vector<weight_type> weights;
};

namespace detail {

// One streaming-level compaction pass produces two groups of items: `retained`
// (at most one carried boundary item that stays at the current level) and
// `promoted` (the kept representatives that move up to the next level). Internal
// helper type for compact_sorted_values_once().
struct CompactValuesResult {
  std::vector<double> retained;
  std::vector<double> promoted;
};

inline void validate_size_limit(std::size_t s) {
  if (s == 0) {
    throw std::invalid_argument("s must be positive");
  }
}

inline void validate_quantile(double q) {
  if (!(q >= 0.0 && q <= 1.0)) {
    throw std::invalid_argument("q must be in [0, 1]");
  }
}

inline bool keep_right_for_pass(CompactMode mode, std::mt19937_64* rng) {
  if (mode == CompactMode::Deterministic) {
    return true;
  }
  if (rng == nullptr) {
    throw std::invalid_argument("random compaction requires an RNG");
  }
  return ((*rng)() & 1ULL) != 0;
}

inline bool carry_odd_item(OddPolicy odd_policy) {
  return odd_policy == OddPolicy::Carry;
}

inline void require_no_nan(double value, const char* name) {
  if (std::isnan(value)) {
    throw std::invalid_argument(std::string(name) + " must not contain NaN");
  }
}

inline bool sample_less(const WeightedSample& a, const WeightedSample& b) {
  return a.value < b.value;
}

// Combine only adjacent equal values. Callers must already have sorted by value.
inline std::vector<WeightedSample> combine_equal_sorted(
    const std::vector<WeightedSample>& samples) {
  if (samples.empty()) {
    return {};
  }

  std::vector<WeightedSample> combined;
  combined.reserve(samples.size());
  WeightedSample cur = samples.front();
  for (std::size_t i = 1; i < samples.size(); ++i) {
    const auto& item = samples[i];
    if (item.value == cur.value) {
      cur.weight += item.weight;
    } else {
      combined.push_back(cur);
      cur = item;
    }
  }
  combined.push_back(cur);
  return combined;
}

inline std::vector<WeightedSample> sorted_combined(
    const std::vector<WeightedSample>& samples) {
  if (samples.empty()) {
    return {};
  }
  std::vector<WeightedSample> out;
  out.reserve(samples.size());
  for (const auto& sample : samples) {
    require_no_nan(sample.value, "samples");
    if (sample.weight > 0) {
      out.push_back(sample);
    }
  }
  std::sort(out.begin(), out.end(), sample_less);
  return combine_equal_sorted(out);
}

// O(n + m) merge of two sorted active item lists. This is the internal path used
// by WeightedSketch1D::merge, where both inputs are already sorted and compact.
inline std::vector<WeightedSample> merge_sorted_weighted(
    const std::vector<WeightedSample>& left,
    const std::vector<WeightedSample>& right,
    bool combine_equal = true) {
  std::vector<WeightedSample> out;
  out.reserve(left.size() + right.size());

  auto push = [&](const WeightedSample& item) {
    if (item.weight == 0) {
      return;
    }
    if (combine_equal && !out.empty() && out.back().value == item.value) {
      out.back().weight += item.weight;
    } else {
      out.push_back(item);
    }
  };

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i].value <= right[j].value) {
      push(left[i++]);
    } else {
      push(right[j++]);
    }
  }
  while (i < left.size()) {
    push(left[i++]);
  }
  while (j < right.size()) {
    push(right[j++]);
  }
  return out;
}

// Repeatedly halves a sorted weighted list until it fits in `s` active entries.
// This is the single shared compaction routine behind every weighted and
// unit-weight build/merge path.
//
// Each pass pairs adjacent items (i, i+1) and keeps one representative carrying
// the pair's combined weight. `keep_right` selects which side survives: it is
// fixed (true) in Deterministic mode and drawn once per pass in Random mode.
//
// An odd-length pass has one unpaired boundary item. With OddPolicy::Carry that
// item is kept unchanged on the same side being kept (the front when keeping
// left, the back when keeping right), so the list stays sorted and total weight
// is preserved. With OddPolicy::DropLargest the largest item is dropped before
// pairing, which sheds its weight.
inline std::vector<WeightedSample> compact_weighted_sorted(
    const std::vector<WeightedSample>& sorted_samples,
    std::size_t s,
    CompactMode mode,
    std::uint64_t seed,
    OddPolicy odd_policy) {
  validate_size_limit(s);

  std::vector<WeightedSample> current = sorted_samples;
  std::mt19937_64 rng(seed);
  std::mt19937_64* rng_ptr = mode == CompactMode::Random ? &rng : nullptr;

  while (current.size() > s) {
    const bool keep_right = keep_right_for_pass(mode, rng_ptr);
    std::vector<WeightedSample> next;
    next.reserve(carry_odd_item(odd_policy) ? (current.size() + 1) / 2
                                            : current.size() / 2);

    std::size_t begin = 0;
    std::size_t end = current.size();
    if ((current.size() & 1U) != 0) {
      if (carry_odd_item(odd_policy)) {
        if (keep_right) {
          --end;
        } else {
          next.push_back(current.front());
          begin = 1;
        }
      } else {
        --end;
      }
    }

    for (std::size_t i = begin; i < end; i += 2) {
      const auto& left = current[i];
      const auto& right = current[i + 1];
      next.push_back(
          keep_right ? WeightedSample{right.value, left.weight + right.weight}
                     : WeightedSample{left.value, left.weight + right.weight});
    }
    if ((current.size() & 1U) != 0 && keep_right &&
        carry_odd_item(odd_policy)) {
      next.push_back(current.back());
    }

    current = combine_equal_sorted(next);
  }

  // Fold any equal values in the final result so the returned representation is
  // canonical even when no compaction pass ran (e.g. unit-weight input already
  // within `s`). This is a no-op when a pass ran or the input was pre-combined.
  return combine_equal_sorted(current);
}

// Convert active items to arrays. Padding is opt-in because most local uses only
// need the retained active items; fixed-shape protocols can request padding.
inline SketchArrays export_arrays(
    const std::vector<WeightedSample>& samples,
    std::size_t s,
    bool fixed_size) {
  validate_size_limit(s);
  if (samples.size() > s) {
    throw std::invalid_argument("cannot export: samples longer than s");
  }

  SketchArrays out;
  out.values.reserve(fixed_size ? s : samples.size());
  out.weights.reserve(fixed_size ? s : samples.size());
  for (const auto& sample : samples) {
    out.values.push_back(sample.value);
    out.weights.push_back(sample.weight);
  }
  if (!fixed_size) {
    return out;
  }
  const std::size_t pad = s - samples.size();
  out.values.insert(
      out.values.end(), pad, std::numeric_limits<double>::infinity());
  out.weights.insert(out.weights.end(), pad, weight_type{0});
  return out;
}

// Import public arrays, ignore zero-weight padding, sort defensively, and merge
// duplicate values.
inline std::vector<WeightedSample> active_items_from_arrays(
    const std::vector<double>& values,
    const std::vector<weight_type>& weights) {
  if (values.size() != weights.size()) {
    throw std::invalid_argument("values and weights must have same length");
  }

  std::vector<WeightedSample> active;
  active.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (weights[i] == 0) {
      continue;
    }
    require_no_nan(values[i], "values");
    active.push_back(WeightedSample{values[i], weights[i]});
  }
  std::sort(active.begin(), active.end(), sample_less);
  return combine_equal_sorted(active);
}

inline bool is_sorted_ascending(const std::vector<double>& values) {
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] < values[i - 1]) {
      return false;
    }
  }
  return true;
}

// Unit-weight entry point. Validates input, sorts only if a cheap scan finds it
// out of order, then delegates to the shared compact_weighted_sorted() so the
// odd-item/compaction logic lives in exactly one place.
//
// Values are intentionally NOT pre-combined here: compaction must pair the raw
// unit-weight items to reproduce KLL's behavior (folding equal values up front
// would change the pairing), and equal values are combined after each pass
// inside compact_weighted_sorted().
inline std::vector<WeightedSample> build_active_from_sorted_values(
    const std::vector<double>& values_sorted,
    std::size_t s,
    CompactMode mode,
    std::uint64_t seed,
    OddPolicy odd_policy) {
  validate_size_limit(s);
  for (double value : values_sorted) {
    require_no_nan(value, "values_sorted");
  }

  std::vector<double> values = values_sorted;
  if (!is_sorted_ascending(values)) {
    std::sort(values.begin(), values.end());
  }

  std::vector<WeightedSample> samples;
  samples.reserve(values.size());
  for (double value : values) {
    samples.push_back(WeightedSample{value, weight_type{1}});
  }
  return compact_weighted_sorted(samples, s, mode, seed, odd_policy);
}

// Streaming-level compaction promotes unweighted sorted values to the next
// level. If a level has odd length, one boundary item remains at the current
// level so the stream's total weight is preserved.
inline CompactValuesResult compact_sorted_values_once(
    const std::vector<double>& sorted_values,
    CompactMode mode,
    std::mt19937_64* rng,
    OddPolicy odd_policy) {
  if (sorted_values.empty()) {
    return {};
  }

  const bool keep_right = keep_right_for_pass(mode, rng);
  CompactValuesResult result;
  result.retained.reserve(1);
  result.promoted.reserve(sorted_values.size() / 2);

  std::size_t begin = 0;
  std::size_t end = sorted_values.size();
  if ((sorted_values.size() & 1U) != 0) {
    if (carry_odd_item(odd_policy)) {
      if (keep_right) {
        --end;
        result.retained.push_back(sorted_values.back());
      } else {
        result.retained.push_back(sorted_values.front());
        begin = 1;
      }
    } else {
      --end;
    }
  }

  for (std::size_t i = begin; i < end; i += 2) {
    result.promoted.push_back(
        keep_right ? sorted_values[i + 1] : sorted_values[i]);
  }
  return result;
}

}  // namespace detail

/**
 * Parse `"deterministic"` or `"random"` into a CompactMode.
 */
inline CompactMode compact_mode_from_string(const std::string& mode) {
  if (mode == "deterministic") {
    return CompactMode::Deterministic;
  }
  if (mode == "random") {
    return CompactMode::Random;
  }
  throw std::invalid_argument("mode must be 'deterministic' or 'random'");
}

/**
 * Return the lowercase string name used by compact_mode_from_string().
 */
inline std::string compact_mode_to_string(CompactMode mode) {
  return mode == CompactMode::Deterministic ? "deterministic" : "random";
}

/**
 * Parse `"carry"` or `"drop_largest"` into an OddPolicy.
 */
inline OddPolicy odd_policy_from_string(const std::string& policy) {
  if (policy == "carry") {
    return OddPolicy::Carry;
  }
  if (policy == "drop_largest") {
    return OddPolicy::DropLargest;
  }
  throw std::invalid_argument("odd policy must be 'carry' or 'drop_largest'");
}

/**
 * Return the lowercase string name used by odd_policy_from_string().
 */
inline std::string odd_policy_to_string(OddPolicy policy) {
  return policy == OddPolicy::Carry ? "carry" : "drop_largest";
}

/**
 * DataSketches-style empirical normalized rank-error estimate for KLL.
 *
 * The returned value is a practical 99%-confidence-style estimate based on the
 * commonly used KLL fit, not a formal guarantee for this minimal implementation.
 */
inline double normalized_rank_error(std::size_t k, bool pmf = false) {
  if (k == 0) {
    throw std::invalid_argument("k must be positive");
  }
  const double coefficient = pmf ? 2.446 : 2.296;
  const double exponent = pmf ? 0.9433 : 0.9723;
  return coefficient / std::pow(static_cast<double>(k), exponent);
}

/**
 * Approximate k needed for a target normalized rank error.
 */
inline std::size_t suggest_k(double epsilon, bool pmf = false) {
  if (!(epsilon > 0.0 && epsilon < 1.0)) {
    throw std::invalid_argument("epsilon must be in (0, 1)");
  }
  const double coefficient = pmf ? 2.446 : 2.296;
  const double exponent = pmf ? 0.9433 : 0.9723;
  return static_cast<std::size_t>(
      std::ceil(std::pow(coefficient / epsilon, 1.0 / exponent)));
}

/**
 * Sorted weighted sketch for one numeric stream.
 *
 * The class stores only active retained items internally. Call `to_arrays()` to
 * obtain active `(values, weights)` arrays, or pass `fixed_size=true` if your
 * downstream use case needs a fixed-size representation.
 */
class WeightedSketch1D {
 public:
  /**
   * Create an empty sketch with capacity/export length `s`.
   */
  explicit WeightedSketch1D(std::size_t s) : s_(s) {
    detail::validate_size_limit(s_);
  }

  /**
   * Build a sketch from unsorted weighted samples.
   *
   * Zero-weight samples are ignored. Duplicate values are combined before
   * compaction. NaN values are rejected because they do not have a strict sort
   * order.
   */
  static WeightedSketch1D from_samples(
      const std::vector<WeightedSample>& samples,
      std::size_t s,
      CompactMode mode = CompactMode::Random,
      std::uint64_t seed = 0,
      OddPolicy odd_policy = OddPolicy::Carry) {
    auto sorted = detail::sorted_combined(samples);
    auto compacted =
        detail::compact_weighted_sorted(sorted, s, mode, seed, odd_policy);
    return WeightedSketch1D(s, std::move(compacted));
  }

  /**
   * Build a sketch from unit-weight values.
   *
   * Input is checked and sorted only if needed.
   */
  static WeightedSketch1D from_sorted_values(
      const std::vector<double>& values_sorted,
      std::size_t s,
      CompactMode mode = CompactMode::Random,
      std::uint64_t seed = 0,
      OddPolicy odd_policy = OddPolicy::Carry) {
    auto active = detail::build_active_from_sorted_values(
        values_sorted, s, mode, seed, odd_policy);
    return WeightedSketch1D(s, std::move(active));
  }

  /**
   * Build from already retained weighted samples without compacting.
   */
  static WeightedSketch1D from_retained_samples(
      const std::vector<WeightedSample>& samples,
      std::size_t size_limit = 0) {
    auto active = detail::sorted_combined(samples);
    if (size_limit == 0) {
      size_limit = active.empty() ? 1 : active.size();
    }
    detail::validate_size_limit(size_limit);
    return WeightedSketch1D(size_limit, std::move(active));
  }

  /**
   * Import an array representation.
   *
   * Zero-weight entries are treated as padding and ignored even if their value
   * is finite. Positive-weight entries are sorted and equal values are combined.
   */
  static WeightedSketch1D from_arrays(
      const std::vector<double>& values,
      const std::vector<weight_type>& weights,
      std::size_t s = 0) {
    if (s == 0) {
      s = values.size();
    }
    detail::validate_size_limit(s);
    auto active = detail::active_items_from_arrays(values, weights);
    return WeightedSketch1D(s, std::move(active));
  }

  /** Fixed-size export target. Retained active items may exceed this value. */
  std::size_t size_limit() const noexcept {
    return s_;
  }

  /** Number of active retained items. */
  std::size_t retained() const noexcept {
    return items_.size();
  }

  /** Total represented weight. */
  weight_type total_weight() const noexcept {
    weight_type total = 0;
    for (const auto& item : items_) {
      total += item.weight;
    }
    return total;
  }

  /** True if there are no active retained items. */
  bool empty() const noexcept {
    return items_.empty();
  }

  /** Sorted active retained items. The returned view excludes padding. */
  const std::vector<WeightedSample>& items() const noexcept {
    return items_;
  }

  /**
   * Return arrays of values and weights.
   *
   * By default only active retained items are returned. Pass `fixed_size=true` if
   * you need exactly `size_limit()` entries, padded with (+inf, 0).
   * Fixed-size export throws if the sketch currently retains more than
   * `size_limit()` active entries; call a fixed-size finalize/merge path first.
   */
  SketchArrays to_arrays(bool fixed_size = false) const {
    return detail::export_arrays(items_, s_, fixed_size);
  }

  /** Convenience wrapper for `to_arrays(true)`. */
  SketchArrays to_padded_arrays() const {
    return to_arrays(true);
  }

  /**
   * Weighted quantile using an inclusive rank rule.
   *
   * The target cumulative weight is `ceil(q * total_weight)`. For `q=0`, this
   * means the first active item is returned. For even unweighted inputs,
   * `q=0.5` returns the lower median, matching Apache DataSketches when queried
   * with `inclusive=true`.
   */
  double quantile(double q) const {
    detail::validate_quantile(q);
    if (items_.empty()) {
      throw std::invalid_argument(
          "quantile undefined for empty sketch (total weight is 0)");
    }

    weight_type total = 0;
    for (const auto& item : items_) {
      total += item.weight;
    }
    if (total == 0) {
      throw std::invalid_argument(
          "quantile undefined for empty sketch (total weight is 0)");
    }

    weight_type target = 0;
    if (q > 0.0) {
      const long double raw =
          static_cast<long double>(q) * static_cast<long double>(total);
      target = static_cast<weight_type>(std::ceil(raw));
    }

    weight_type prefix = 0;
    for (const auto& item : items_) {
      prefix += item.weight;
      if (prefix >= target) {
        return item.value;
      }
    }
    return items_.back().value;
  }

  /**
   * Batch quantile query.
   *
   * This computes total weight once, sorts requested ranks internally, and
   * scans retained items once. The returned values preserve the caller's input
   * order.
   */
  std::vector<double> quantiles(const std::vector<double>& qs) const {
    if (qs.empty()) {
      return {};
    }
    if (items_.empty()) {
      throw std::invalid_argument(
          "quantile undefined for empty sketch (total weight is 0)");
    }

    weight_type total = 0;
    for (const auto& item : items_) {
      total += item.weight;
    }
    if (total == 0) {
      throw std::invalid_argument(
          "quantile undefined for empty sketch (total weight is 0)");
    }

    struct Query {
      weight_type target = 0;
      std::size_t index = 0;
    };

    std::vector<Query> queries;
    queries.reserve(qs.size());
    for (std::size_t i = 0; i < qs.size(); ++i) {
      detail::validate_quantile(qs[i]);
      weight_type target = 0;
      if (qs[i] > 0.0) {
        const long double raw =
            static_cast<long double>(qs[i]) * static_cast<long double>(total);
        target = static_cast<weight_type>(std::ceil(raw));
      }
      queries.push_back(Query{target, i});
    }

    std::sort(
        queries.begin(), queries.end(), [](const Query& a, const Query& b) {
          if (a.target != b.target) {
            return a.target < b.target;
          }
          return a.index < b.index;
        });

    std::vector<double> out(qs.size());
    std::size_t query_idx = 0;
    weight_type prefix = 0;
    for (const auto& item : items_) {
      prefix += item.weight;
      while (query_idx < queries.size() && prefix >= queries[query_idx].target) {
        out[queries[query_idx].index] = item.value;
        ++query_idx;
      }
    }

    while (query_idx < queries.size()) {
      out[queries[query_idx].index] = items_.back().value;
      ++query_idx;
    }

    return out;
  }

  /** Shortcut for `quantile(0.5)`. */
  double median() const {
    return quantile(0.5);
  }

  /**
   * Merge two sketches with the same export length and compact back to `s`.
   */
  WeightedSketch1D merge(
      const WeightedSketch1D& other,
      CompactMode mode = CompactMode::Random,
      std::uint64_t seed = 0,
      bool fixed_size = false,
      OddPolicy odd_policy = OddPolicy::Carry) const {
    if (s_ != other.s_) {
      throw std::invalid_argument("cannot merge sketches with different s");
    }
    auto merged = detail::merge_sorted_weighted(items_, other.items_);
    if (fixed_size) {
      auto compacted =
          detail::compact_weighted_sorted(merged, s_, mode, seed, odd_policy);
      return WeightedSketch1D(s_, std::move(compacted));
    }
    return WeightedSketch1D(s_, std::move(merged));
  }

 private:
  WeightedSketch1D(std::size_t s, std::vector<WeightedSample>&& items)
      : s_(s), items_(std::move(items)) {
    detail::validate_size_limit(s_);
  }

  std::size_t s_ = 0;
  std::vector<WeightedSample> items_;
};

/**
 * Build arrays from sorted unit-weight values.
 *
 * Fixed-size export is disabled by default. Pass `fixed_size=true` if you need exactly `s`
 * entries.
 */
inline SketchArrays build_sketch_from_sorted_values(
    const std::vector<double>& values_sorted,
    std::size_t s,
    CompactMode mode = CompactMode::Random,
    std::uint64_t seed = 0,
    bool fixed_size = false,
    OddPolicy odd_policy = OddPolicy::Carry) {
  const auto active = detail::build_active_from_sorted_values(
      values_sorted, s, mode, seed, odd_policy);
  return detail::export_arrays(active, s, fixed_size);
}

/**
 * Build arrays from one column.
 *
 * The column is sorted only if a cheap O(n) scan finds it out of order, so an
 * already-sorted column incurs no extra sort. `pre_sorted` is a caller-facing
 * intent hint only: sortedness is always verified, so a mislabeled column still
 * produces correct output. This matches `KLLSketch1D::update_batch`.
 * Fixed-size export is disabled by default. Pass `fixed_size=true` if you need
 * exactly `s` entries.
 */
inline SketchArrays build_sketch_from_column(
    std::vector<double> column,
    std::size_t s,
    bool pre_sorted = false,
    CompactMode mode = CompactMode::Random,
    std::uint64_t seed = 0,
    bool fixed_size = false,
    OddPolicy odd_policy = OddPolicy::Carry) {
  (void)pre_sorted;  // Intentional: an intent hint that never affects correctness.
  for (double value : column) {
    detail::require_no_nan(value, "column");
  }
  if (!detail::is_sorted_ascending(column)) {
    std::sort(column.begin(), column.end());
  }
  return build_sketch_from_sorted_values(
      column, s, mode, seed, fixed_size, odd_policy);
}

/**
 * Merge two sorted weighted arrays with an O(n + m) pass.
 *
 * `drop_zero=true` skips padding entries. `combine_equal=true` folds adjacent
 * equal values in the merged output.
 */
inline SketchArrays merge_two_sorted_weighted(
    const std::vector<double>& a_values,
    const std::vector<weight_type>& a_weights,
    const std::vector<double>& b_values,
    const std::vector<weight_type>& b_weights,
    bool drop_zero = true,
    bool combine_equal = false) {
  if (a_values.size() != a_weights.size() ||
      b_values.size() != b_weights.size()) {
    throw std::invalid_argument("values and weights must have same length");
  }

  SketchArrays out;
  out.values.reserve(a_values.size() + b_values.size());
  out.weights.reserve(a_weights.size() + b_weights.size());

  auto push = [&](double value, weight_type weight) {
    if (drop_zero && weight == 0) {
      return;
    }
    if (weight > 0) {
      detail::require_no_nan(value, "values");
    }
    if (combine_equal && !out.values.empty() && out.values.back() == value) {
      out.weights.back() += weight;
    } else {
      out.values.push_back(value);
      out.weights.push_back(weight);
    }
  };

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < a_values.size() && j < b_values.size()) {
    if (a_values[i] <= b_values[j]) {
      push(a_values[i], a_weights[i]);
      ++i;
    } else {
      push(b_values[j], b_weights[j]);
      ++j;
    }
  }
  while (i < a_values.size()) {
    push(a_values[i], a_weights[i]);
    ++i;
  }
  while (j < b_values.size()) {
    push(b_values[j], b_weights[j]);
    ++j;
  }

  return out;
}

/**
 * Merge many sorted weighted arrays.
 *
 * This implementation uses a heap-based K-way merge instead of repeated
 * pairwise merges, reducing work for many inputs.
 */
inline SketchArrays merge_k_sorted_lists(
    const std::vector<std::vector<double>>& values_list,
    const std::vector<std::vector<weight_type>>& weights_list,
    bool drop_zero = true,
    bool combine_equal = false) {
  if (values_list.size() != weights_list.size()) {
    throw std::invalid_argument("vals_list and wts_list must have same length");
  }
  if (values_list.empty()) {
    throw std::invalid_argument("expected at least one sorted list");
  }
  for (std::size_t list = 0; list < values_list.size(); ++list) {
    if (values_list[list].size() != weights_list[list].size()) {
      throw std::invalid_argument("values and weights must have same length");
    }
  }

  struct HeapItem {
    double value;
    weight_type weight;
    std::size_t list;
    std::size_t pos;
  };
  struct Greater {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
      if (a.value != b.value) {
        return a.value > b.value;
      }
      if (a.list != b.list) {
        return a.list > b.list;
      }
      return a.pos > b.pos;
    }
  };

  auto next_pos = [&](std::size_t list, std::size_t pos) {
    while (pos < values_list[list].size() && drop_zero &&
           weights_list[list][pos] == 0) {
      ++pos;
    }
    return pos;
  };

  std::priority_queue<HeapItem, std::vector<HeapItem>, Greater> heap;
  std::size_t reserve = 0;
  for (std::size_t list = 0; list < values_list.size(); ++list) {
    reserve += values_list[list].size();
    const std::size_t pos = next_pos(list, 0);
    if (pos < values_list[list].size()) {
      heap.push(HeapItem{
          values_list[list][pos], weights_list[list][pos], list, pos});
    }
  }

  SketchArrays out;
  out.values.reserve(reserve);
  out.weights.reserve(reserve);

  auto push = [&](double value, weight_type weight) {
    if (drop_zero && weight == 0) {
      return;
    }
    if (weight > 0) {
      detail::require_no_nan(value, "values");
    }
    if (combine_equal && !out.values.empty() && out.values.back() == value) {
      out.weights.back() += weight;
    } else {
      out.values.push_back(value);
      out.weights.push_back(weight);
    }
  };

  while (!heap.empty()) {
    const HeapItem item = heap.top();
    heap.pop();
    push(item.value, item.weight);

    const std::size_t pos = next_pos(item.list, item.pos + 1);
    if (pos < values_list[item.list].size()) {
      heap.push(HeapItem{
          values_list[item.list][pos],
          weights_list[item.list][pos],
          item.list,
          pos});
    }
  }

  return out;
}

/**
 * Weighted median over sorted or unsorted weighted arrays.
 *
 * Zero-weight entries are ignored. The target rank is `(total_weight + 1) / 2`,
 * which is the same lower-median rule used by `WeightedSketch1D::median()`.
 */
inline double weighted_median_from_sorted_weighted(
    const std::vector<double>& values,
    const std::vector<weight_type>& weights) {
  if (values.size() != weights.size()) {
    throw std::invalid_argument("values and weights must have same length");
  }

  std::vector<WeightedSample> active;
  active.reserve(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (weights[i] == 0) {
      continue;
    }
    detail::require_no_nan(values[i], "values");
    active.push_back(WeightedSample{values[i], weights[i]});
  }
  if (active.empty()) {
    throw std::invalid_argument(
        "weighted median undefined for empty weighted list");
  }
  if (!std::is_sorted(active.begin(), active.end(), detail::sample_less)) {
    std::sort(active.begin(), active.end(), detail::sample_less);
  }

  weight_type total = 0;
  for (const auto& item : active) {
    total += item.weight;
  }
  if (total == 0) {
    throw std::invalid_argument(
        "weighted median undefined for non-positive total weight");
  }

  const weight_type target = (total + 1) / 2;
  weight_type prefix = 0;
  for (const auto& item : active) {
    prefix += item.weight;
    if (prefix >= target) {
      return item.value;
    }
  }
  return active.back().value;
}

/**
 * Merge many sorted weighted sketches and return their weighted median.
 */
inline double weighted_median_from_k_sorted_sketches(
    const std::vector<std::vector<double>>& values_list,
    const std::vector<std::vector<weight_type>>& weights_list,
    bool drop_zero = true,
    bool combine_equal = false) {
  const auto merged =
      merge_k_sorted_lists(values_list, weights_list, drop_zero, combine_equal);
  return weighted_median_from_sorted_weighted(merged.values, merged.weights);
}

/**
 * Median helper for a single fixed array representation.
 */
inline double weighted_median_from_arrays(
    const std::vector<double>& values,
    const std::vector<weight_type>& weights) {
  return WeightedSketch1D::from_arrays(values, weights).median();
}

/**
 * Streaming 1D builder.
 *
 * Updates are buffered at level 0. Whenever a level exceeds `k`, it is sorted,
 * compacted by half, and promoted to the next level. Finalization converts each
 * retained level item into a WeightedSample with weight `2^level`, then compacts
 * to the requested export length `s`.
 */
class KLLSketch1D {
 public:
  /**
   * Create a streaming sketch.
   *
   * `k` controls when individual levels compact. `s` controls the final
   * exported sketch size.
   */
  KLLSketch1D(
      std::size_t k,
      std::size_t s,
      CompactMode mode = CompactMode::Random,
      std::uint64_t seed = 0,
      bool ignore_nan = true,
      OddPolicy odd_policy = OddPolicy::Carry)
      : k_(k),
        s_(s),
        mode_(mode),
        seed_(seed),
        ignore_nan_(ignore_nan),
        odd_policy_(odd_policy),
        rng_(seed) {
    if (k_ == 0) {
      throw std::invalid_argument("k must be positive");
    }
    detail::validate_size_limit(s_);
    levels_.emplace_back();
  }

  /** Insert one sample. NaNs are ignored or rejected according to ignore_nan. */
  void update(double x) {
    if (std::isnan(x)) {
      if (ignore_nan_) {
        return;
      }
      throw std::invalid_argument("NaN encountered and ignore_nan=false");
    }

    levels_[0].push_back(x);
    ++count_;
    compact_level(0);
  }

  /** Insert all samples from a vector. */
  void update_many(const std::vector<double>& xs) {
    for (double x : xs) {
      update(x);
    }
  }

  /**
   * Insert a materialized batch with one level-0 append.
   *
   * This path filters or rejects NaNs once, sorts the batch only if needed, and
   * compacts after appending the whole batch. It is usually faster than calling
   * `update()` for every value when the input vector is already in memory.
   *
   * The resulting sketch is valid but may not be bit-identical to `update_many`
   * because the compaction boundaries are different.
   */
  void update_batch(std::vector<double> xs, bool pre_sorted = false) {
    if (xs.empty()) {
      return;
    }

    std::vector<double> filtered;
    filtered.reserve(xs.size());
    for (double x : xs) {
      if (std::isnan(x)) {
        if (ignore_nan_) {
          continue;
        }
        throw std::invalid_argument("NaN encountered and ignore_nan=false");
      }
      filtered.push_back(x);
    }
    if (filtered.empty()) {
      return;
    }
    const std::size_t accepted = filtered.size();

    // `pre_sorted` is a caller-facing intent hint only. Sortedness is always
    // verified with a cheap O(n) scan and the batch is sorted only if needed, so
    // a mislabeled batch can never produce an unsorted (mis-compacted) level.
    // Mirrors build_sketch_from_column().
    (void)pre_sorted;
    if (!detail::is_sorted_ascending(filtered)) {
      std::sort(filtered.begin(), filtered.end());
    }

    const bool level0_was_empty = levels_[0].empty();
    if (level0_was_empty) {
      levels_[0] = std::move(filtered);
    } else {
      levels_[0].insert(levels_[0].end(), filtered.begin(), filtered.end());
    }
    count_ += accepted;
    compact_level(0, level0_was_empty);
  }

  /**
   * Convenience wrapper for `update_batch(xs, true)`.
   *
   * Input is still checked and sorted if needed; the name documents the fast
   * path callers should prefer when they already have a sorted column.
   */
  void update_sorted_batch(std::vector<double> xs) {
    update_batch(std::move(xs), true);
  }

  /**
   * Export the current stream state.
   *
   * By default this returns all retained level items without an extra final
   * compaction pass. Pass `fixed_size=true` to compact once to `s` retained
   * items for fixed-shape downstream use.
   */
  WeightedSketch1D finalize(bool fixed_size = false) const {
    std::vector<WeightedSample> samples;
    std::size_t retained = 0;
    for (const auto& level : levels_) {
      retained += level.size();
    }
    samples.reserve(retained);

    for (std::size_t level = 0; level < levels_.size(); ++level) {
      if (levels_[level].empty()) {
        continue;
      }
      if (level >= 64) {
        throw std::overflow_error("level weight exceeds uint64 range");
      }
      const weight_type weight = weight_type{1} << level;
      for (double value : levels_[level]) {
        samples.push_back(WeightedSample{value, weight});
      }
    }

    if (fixed_size) {
      return WeightedSketch1D::from_samples(
          samples, s_, mode_, seed_, odd_policy_);
    }
    auto active = detail::sorted_combined(samples);
    return WeightedSketch1D::from_retained_samples(active, s_);
  }

  /** Number of non-NaN updates accepted by the sketch. */
  std::size_t count() const noexcept {
    return count_;
  }

  /** Retained values by level, exposed for diagnostics and tests. */
  const std::vector<std::vector<double>>& levels() const noexcept {
    return levels_;
  }

  /** Configured level capacity. */
  std::size_t k() const noexcept {
    return k_;
  }

  /**
   * Configured export size `s` (the target retained-item count used by
   * fixed-size finalize). Named to match `WeightedSketch1D::size_limit()`.
   */
  std::size_t size_limit() const noexcept {
    return s_;
  }

  /** Empirical normalized rank-error estimate for this sketch's k. */
  double normalized_rank_error(bool pmf = false) const {
    return minikll::normalized_rank_error(k_, pmf);
  }

 private:
  // Cascades compaction upward. Promotion can overflow the next level, so this
  // loop continues until every touched level is within capacity.
  void compact_level(std::size_t level, bool current_level_sorted = false) {
    while (level < levels_.size() && levels_[level].size() > k_) {
      auto& current = levels_[level];
      if (!current_level_sorted) {
        std::sort(current.begin(), current.end());
      }
      auto compacted =
          detail::compact_sorted_values_once(current, mode_, &rng_, odd_policy_);
      current = std::move(compacted.retained);

      const std::size_t next_level = level + 1;
      if (next_level == levels_.size()) {
        levels_.emplace_back();
      }
      auto& next = levels_[next_level];
      next.insert(
          next.end(), compacted.promoted.begin(), compacted.promoted.end());
      level = next_level;
      current_level_sorted = false;
    }
  }

  std::size_t k_ = 0;
  std::size_t s_ = 0;
  CompactMode mode_ = CompactMode::Random;
  std::uint64_t seed_ = 0;
  bool ignore_nan_ = true;
  OddPolicy odd_policy_ = OddPolicy::Carry;
  std::vector<std::vector<double>> levels_;
  std::size_t count_ = 0;
  std::mt19937_64 rng_;
};

/**
 * Column-wise wrapper around KLLSketch1D.
 */
class KLLSketchND {
 public:
  /**
   * Create `d` independent column sketches.
   *
   * Random mode uses `seed + column_index` to avoid identical compaction streams
   * across dimensions.
   */
  KLLSketchND(
      std::size_t k,
      std::size_t s,
      std::size_t d,
      CompactMode mode = CompactMode::Random,
      std::uint64_t seed = 0,
      bool ignore_nan = true,
      OddPolicy odd_policy = OddPolicy::Carry)
      : d_(d) {
    if (d_ == 0) {
      throw std::invalid_argument("d must be positive");
    }
    columns_.reserve(d_);
    for (std::size_t i = 0; i < d_; ++i) {
      columns_.emplace_back(k, s, mode, seed + i, ignore_nan, odd_policy);
    }
  }

  /** Update every column sketch from one row. */
  void update_row(const std::vector<double>& row) {
    if (row.size() != d_) {
      throw std::invalid_argument("row length does not match d");
    }
    for (std::size_t i = 0; i < d_; ++i) {
      columns_[i].update(row[i]);
    }
  }

  /** Update from a row-major matrix. */
  void update_matrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
      update_row(row);
    }
  }

  /** Finalize each column sketch. */
  std::vector<WeightedSketch1D> finalize(bool fixed_size = false) const {
    std::vector<WeightedSketch1D> out;
    out.reserve(columns_.size());
    for (const auto& column : columns_) {
      out.push_back(column.finalize(fixed_size));
    }
    return out;
  }

  /** Number of columns. */
  std::size_t dimension() const noexcept {
    return d_;
  }

 private:
  std::size_t d_ = 0;
  std::vector<KLLSketch1D> columns_;
};

/**
 * Merge many WeightedSketch1D instances by collecting all active items and
 * compacting at most once.
 */
inline WeightedSketch1D merge_weighted_sketches(
    const std::vector<WeightedSketch1D>& sketches,
    CompactMode mode = CompactMode::Random,
    std::uint64_t seed = 0,
    bool fixed_size = false,
    OddPolicy odd_policy = OddPolicy::Carry) {
  if (sketches.empty()) {
    throw std::invalid_argument("sketches must be non-empty");
  }

  const std::size_t s = sketches.front().size_limit();
  for (const auto& sketch : sketches) {
    if (sketch.size_limit() != s) {
      throw std::invalid_argument("all sketches must have the same s");
    }
  }

  std::size_t total_retained = 0;
  for (const auto& sketch : sketches) {
    total_retained += sketch.retained();
  }
  std::vector<WeightedSample> all;
  all.reserve(total_retained);
  for (const auto& sketch : sketches) {
    for (const auto& item : sketch.items()) {
      all.push_back(item);
    }
  }

  auto active = detail::sorted_combined(all);
  if (fixed_size) {
    auto compacted =
        detail::compact_weighted_sorted(active, s, mode, seed, odd_policy);
    return WeightedSketch1D::from_retained_samples(compacted, s);
  }
  return WeightedSketch1D::from_retained_samples(active, s);
}

}  // namespace minikll

#endif  // MINIKLL_MINIKLL_HPP_
